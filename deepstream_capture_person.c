/*
 * DeepStream application to capture images when person detected
 * Based on deepstream_proximity_yolo.c
 *
 * Compile with:
 *   gcc -o deepstream_capture_person deepstream_capture_person.c \
 *       $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-video-1.0) \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta \
 *       -lnvbufsurface -lnvbufsurftransform -ljpeg -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib
 */

#include <gst/gst.h>
#include <gst/video/video.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <jpeglib.h>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#define PERSON_CLASS_ID 0
#define MAX_PATH_LEN 256

/* Configuration */
static char output_dir[MAX_PATH_LEN] = "dataset";
static float capture_interval = 1.0;
static int sensor_id = 0;
static int camera_width = 1920;
static int camera_height = 1080;
static int mux_width = 640;
static int mux_height = 360;

/* State */
static double last_capture_time = 0;
static int capture_count = 0;

/* Get current time in seconds */
static double get_current_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Save RGBA buffer as JPEG */
static int save_frame_as_jpeg(NvBufSurface *surface, int batch_id, const char *filename) {
    NvBufSurfaceParams *params = &surface->surfaceList[batch_id];

    /* Map surface for CPU access */
    if (NvBufSurfaceMap(surface, batch_id, -1, NVBUF_MAP_READ) != 0) {
        g_printerr("Failed to map buffer\n");
        return -1;
    }
    NvBufSurfaceSyncForCpu(surface, batch_id, -1);

    int width = params->width;
    int height = params->height;
    int pitch = params->pitch;
    unsigned char *data = (unsigned char *)params->mappedAddr.addr[0];

    /* Open file */
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        g_printerr("Cannot open %s for writing\n", filename);
        NvBufSurfaceUnMap(surface, batch_id, -1);
        return -1;
    }

    /* Setup JPEG compression */
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 90, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    /* Allocate row buffer for RGB conversion */
    unsigned char *row_buffer = (unsigned char *)malloc(width * 3);

    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char *src_row = data + cinfo.next_scanline * pitch;

        /* Convert RGBA to RGB */
        for (int x = 0; x < width; x++) {
            row_buffer[x * 3 + 0] = src_row[x * 4 + 0];  /* R */
            row_buffer[x * 3 + 1] = src_row[x * 4 + 1];  /* G */
            row_buffer[x * 3 + 2] = src_row[x * 4 + 2];  /* B */
        }

        JSAMPROW row_pointer = row_buffer;
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    free(row_buffer);
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);

    NvBufSurfaceUnMap(surface, batch_id, -1);
    return 0;
}

/* OSD buffer probe to detect persons and capture images */
static GstPadProbeReturn
osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);

        int person_count = 0;

        /* Count persons in frame */
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            if (obj_meta->class_id == PERSON_CLASS_ID) {
                person_count++;
            }
        }

        /* Capture if person detected and interval passed */
        if (person_count > 0) {
            double current_time = get_current_time();

            if ((current_time - last_capture_time) >= capture_interval) {
                /* Get surface from buffer */
                GstMapInfo map_info;
                if (gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
                    NvBufSurface *surface = (NvBufSurface *)map_info.data;

                    /* Generate filename */
                    time_t now = time(NULL);
                    struct tm *t = localtime(&now);
                    char filename[MAX_PATH_LEN];
                    snprintf(filename, MAX_PATH_LEN,
                             "%s/person_%04d%02d%02d_%02d%02d%02d_%04d.jpg",
                             output_dir,
                             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
                             t->tm_hour, t->tm_min, t->tm_sec,
                             capture_count);

                    /* Save frame */
                    if (save_frame_as_jpeg(surface, frame_meta->batch_id, filename) == 0) {
                        capture_count++;
                        last_capture_time = current_time;
                        g_print("[%d] Captured: %s (%d person(s))\n",
                                capture_count, filename, person_count);
                    }

                    gst_buffer_unmap(buf, &map_info);
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME(msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

static void print_usage(const char *prog) {
    g_print("Usage: %s [OPTIONS]\n\n", prog);
    g_print("Options:\n");
    g_print("  -o, --output-dir DIR    Output directory (default: dataset)\n");
    g_print("  -i, --interval SEC      Capture interval in seconds (default: 1.0)\n");
    g_print("  -s, --sensor-id ID      CSI sensor ID (default: 0)\n");
    g_print("  -c, --config FILE       Inference config file (default: config_infer_yolov8n-seg.txt)\n");
    g_print("  -h, --help              Show this help\n");
    g_print("\nExamples:\n");
    g_print("  %s -o faces -i 2.0\n", prog);
    g_print("  %s -s 1 -c config_infer_yolo8s_single.txt\n", prog);
}

int main(int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *caps_filter = NULL,
               *nvvidconv_src = NULL, *streammux = NULL, *nvinfer = NULL,
               *nvvidconv = NULL, *nvosd = NULL, *nvvidconv2 = NULL, *sink = NULL;
    GstCaps *caps = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    char config_file[MAX_PATH_LEN] = "config_infer_yolov8n-seg.txt";

    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-dir") == 0) && i + 1 < argc) {
            strncpy(output_dir, argv[++i], MAX_PATH_LEN - 1);
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interval") == 0) && i + 1 < argc) {
            capture_interval = atof(argv[++i]);
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sensor-id") == 0) && i + 1 < argc) {
            sensor_id = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) && i + 1 < argc) {
            strncpy(config_file, argv[++i], MAX_PATH_LEN - 1);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* Create output directory */
    mkdir(output_dir, 0755);

    g_print("DeepStream Person Capture\n");
    g_print("========================================\n");
    g_print("Output: %s\n", output_dir);
    g_print("Interval: %.1fs\n", capture_interval);
    g_print("Sensor ID: %d\n", sensor_id);
    g_print("Config: %s\n", config_file);
    g_print("========================================\n");
    g_print("Press Ctrl+C to stop\n\n");

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);

    /* Create elements */
    pipeline = gst_pipeline_new("deepstream-capture-pipeline");
    source = gst_element_factory_make("nvarguscamerasrc", "source");
    caps_filter = gst_element_factory_make("capsfilter", "caps-filter");
    nvvidconv_src = gst_element_factory_make("nvvideoconvert", "nvvidconv-src");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    nvinfer = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter2");
    sink = gst_element_factory_make("nv3dsink", "nvvideo-renderer");

    if (!pipeline || !source || !caps_filter || !nvvidconv_src || !streammux ||
        !nvinfer || !nvvidconv || !nvosd || !nvvidconv2 || !sink) {
        g_printerr("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Configure source */
    g_object_set(G_OBJECT(source), "sensor-id", sensor_id, NULL);
    g_object_set(G_OBJECT(source), "bufapi-version", TRUE, NULL);

    /* Set caps */
    caps = gst_caps_from_string(
        g_strdup_printf("video/x-raw(memory:NVMM), width=%d, height=%d, format=NV12, framerate=30/1",
                        camera_width, camera_height));
    g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
    gst_caps_unref(caps);

    /* Configure streammux */
    g_object_set(G_OBJECT(streammux),
                 "live-source", 1,
                 "width", mux_width,
                 "height", mux_height,
                 "batch-size", 1,
                 "batched-push-timeout", 40000, NULL);

    /* Configure inference */
    g_object_set(G_OBJECT(nvinfer), "config-file-path", config_file, NULL);

    /* Configure sink */
    g_object_set(G_OBJECT(sink), "sync", 0, NULL);

    /* Add elements to pipeline */
    gst_bin_add_many(GST_BIN(pipeline), source, caps_filter, nvvidconv_src,
                     streammux, nvinfer, nvvidconv, nvosd, nvvidconv2, sink, NULL);

    /* Link source -> caps_filter -> nvvidconv_src */
    if (!gst_element_link_many(source, caps_filter, nvvidconv_src, NULL)) {
        g_printerr("Failed to link source elements. Exiting.\n");
        return -1;
    }

    /* Link nvvidconv_src to streammux */
    GstPad *sinkpad, *srcpad;
    sinkpad = gst_element_get_request_pad(streammux, "sink_0");
    srcpad = gst_element_get_static_pad(nvvidconv_src, "src");

    if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link nvvidconv_src to streammux\n");
        return -1;
    }
    gst_object_unref(sinkpad);
    gst_object_unref(srcpad);

    /* Link rest of pipeline */
    if (!gst_element_link_many(streammux, nvinfer, nvvidconv, nvosd, nvvidconv2, sink, NULL)) {
        g_printerr("Elements could not be linked. Exiting.\n");
        return -1;
    }

    /* Add probe to OSD sink pad for capturing */
    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad) {
        g_printerr("Unable to get sink pad\n");
    } else {
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          osd_sink_pad_buffer_probe, NULL, NULL);
    }
    gst_object_unref(osd_sink_pad);

    /* Add bus watch */
    bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    /* Start playing */
    g_print("Starting pipeline...\n");
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_main_loop_run(loop);

    /* Cleanup */
    g_print("\nStopping...\n");
    gst_element_set_state(pipeline, GST_STATE_NULL);
    g_print("Total images captured: %d\n", capture_count);
    g_print("Images saved to: %s\n", output_dir);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
