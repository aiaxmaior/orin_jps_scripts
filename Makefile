CC = gcc
CFLAGS = -Wall `pkg-config --cflags gstreamer-1.0`
LIBS = `pkg-config --libs gstreamer-1.0` -L/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lm -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib
INC = -I/opt/nvidia/deepstream/deepstream/sources/includes

all: deepstream_proximity deepstream_proximity_yolo

deepstream_proximity: deepstream_proximity.c
	$(CC) $(CFLAGS) $(INC) -o deepstream_proximity deepstream_proximity.c $(LIBS)

deepstream_proximity_yolo: deepstream_proximity_yolo.c
	$(CC) $(CFLAGS) $(INC) -o deepstream_proximity_yolo deepstream_proximity_yolo.c $(LIBS)

clean:
	rm -f deepstream_proximity deepstream_proximity_yolo
