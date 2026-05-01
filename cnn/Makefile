CC = gcc
CFLAGS = -Wall -Wextra -g -Ilayers -I.
LDFLAGS = -lm
SRCS = app.c tensor.c layers/conv.c layers/activation.c layers/pool.c layers/fc.c
OBJS = $(SRCS:.c=.o)
TARGET = app

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
