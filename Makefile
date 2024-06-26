NAME := llama2

SRC_DIR := src
OBJ_DIR := build
SRCS    := llama2.c utilities.c
SRCS    := $(SRCS:%=$(SRC_DIR)/%)
OBJS    := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

CC       := gcc
CFLAGS   := -Wall -Wextra -pthread -lm -O2
CPPFLAGS := -I include

MAKEFLAGS += --no-print-directory


all: $(NAME)

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(NAME)
	$(info CREATED $(NAME))

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<
	$(info CREATED $@)

clean:
	rm -rf $(OBJ_DIR) $(NAME)

remake:
	$(MAKE) clean
	$(MAKE) all

.PHONY: cleane remake
# .SILENT:
