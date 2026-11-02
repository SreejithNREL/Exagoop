CC  = gcc
SRC = wall_vel_twist.c

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  TARGET = libwall_twist.dylib
  CFLAGS = -O2 -fPIC -dynamiclib
else
  TARGET = libwall_twist.so
  CFLAGS = -O2 -fPIC -shared
endif

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f libwall_twist.so libwall_twist.dylib
