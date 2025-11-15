/*
 * DeepStream application with proximity distance overlay
 * Based on deepstream-test2
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include "gstnvdsmeta.h"

#define MAX_DISPLAY_LEN 64

/* Proximity sensor parameters */
#define FOCAL_LENGTH 700.0
#define CAMERA_HEIGHT 1.2
#define IMAGE_HEIGHT 360.0

/* Known object heights (meters) */
#define CAR_HEIGHT 1.5
#define PERSON_HEIGHT 1.7
#define BICYCLE_HEIGHT 1.2
#define SIGN_HEIGHT 1.0

/* Calculate distance using pinhole camera model */
static float calculate_distance(float bbox_height, float real_height) {
    if (bbox_height < 1.0) return -1.0;
    return (real_height * FOCAL_LENGTH) / bbox_height;
}

/* Calculate distance using ground plane geometry */
static float calculate_distance_ground_plane(float bbox_bottom_y) {
    float y_diff = IMAGE_HEIGHT - bbox_bottom_y;
    if (y_diff < 1.0) return -1.0;
    return (CAMERA_HEIGHT * FOCAL_LENGTH) / y_diff;
}

/* OSD buffer probe to add distance information */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    NvDsObjectMeta *obj_meta = NULL;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);

            /* Calculate bbox dimensions */
            float bbox_width = obj_meta->rect_params.width;
            float bbox_height = obj_meta->rect_params.height;
            float bbox_bottom_y = obj_meta->rect_params.top + bbox_height;

            /* Determine real height based on class */
            float real_height = 1.5;  // default
            if (obj_meta->class_id == 0) real_height = CAR_HEIGHT;
            else if (obj_meta->class_id == 1) real_height = BICYCLE_HEIGHT;
            else if (obj_meta->class_id == 2) real_height = PERSON_HEIGHT;
            else if (obj_meta->class_id == 3) real_height = SIGN_HEIGHT;

            /* Calculate distance */
            float distance = calculate_distance(bbox_height, real_height);
            float distance_gp = calculate_distance_ground_plane(bbox_bottom_y);

            /* Use average of both methods */
            float final_distance = (distance + distance_gp) / 2.0;

            /* Set color based on distance */
            if (final_distance < 3.0) {
                /* CRITICAL - Red */
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_width = 5;
            } else if (final_distance < 10.0) {
                /* WARNING - Orange */
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 0.65;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_width = 4;
            } else if (final_distance < 20.0) {
                /* CAUTION - Yellow */
                obj_meta->rect_params.border_color.red = 1.0;
                obj_meta->rect_params.border_color.green = 1.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_width = 3;
            } else {
                /* SAFE - Green */
                obj_meta->rect_params.border_color.red = 0.0;
                obj_meta->rect_params.border_color.green = 1.0;
                obj_meta->rect_params.border_color.blue = 0.0;
                obj_meta->rect_params.border_width = 3;
            }

            /* Update display text with distance */
            if (obj_meta->text_params.display_text) {
                snprintf(obj_meta->text_params.display_text, MAX_DISPLAY_LEN,
                         "%s %.1fm", obj_meta->obj_label, final_distance);
            }

            /* Print critical warnings */
            if (final_distance < 3.0) {
                g_print("[CRITICAL] %s at %.1fm\n", obj_meta->obj_label, final_distance);
            }
        }
    }
    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr ("Error details: %s\n", debug);
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int
main (int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *nvvidconv = NULL,
               *nvinfer = NULL, *nvvidconv2 = NULL, *nvosd = NULL, *sink = NULL,
               *streammux = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create elements */
    pipeline = gst_pipeline_new ("deepstream-proximity-pipeline");
    source = gst_element_factory_make ("nvarguscamerasrc", "source");
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
    nvinfer = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvvidconv2 = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter2");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make ("nv3dsink", "nvvideo-renderer");

    if (!pipeline || !source || !streammux || !nvinfer || !nvvidconv || 
        !nvvidconv2 || !nvosd || !sink) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Set properties */
    g_object_set (G_OBJECT (source), "sensor-id", 0, NULL);
    g_object_set (G_OBJECT (streammux), "width", 640, "height", 360,
                  "batch-size", 1, "batched-push-timeout", 40000, NULL);
    g_object_set (G_OBJECT (nvinfer),
                  "config-file-path", "config_infer_dashcamnet_single.txt", NULL);
    g_object_set (G_OBJECT (sink), "sync", 0, NULL);

    /* Add all elements to pipeline */
    gst_bin_add_many (GST_BIN (pipeline), source, streammux, nvinfer,
                      nvvidconv, nvvidconv2, nvosd, sink, NULL);

    /* Link elements */
    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    srcpad = gst_element_get_static_pad (source, "src");
    
    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link source to streammux\n");
        return -1;
    }
    
    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    if (!gst_element_link_many (streammux, nvinfer, nvvidconv, nvvidconv2,
                                nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
    }

    /* Add probe to OSD sink pad */
    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                           osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref (osd_sink_pad);

    /* Add bus watch */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Start playing */
    g_print ("Running DeepStream with proximity sensor...\n");
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    g_main_loop_run (loop);

    /* Cleanup */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}
