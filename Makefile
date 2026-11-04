# Convenience wrappers around idf.py / esptool. Settings come from an optional .env file.
-include .env
export

# Model file produced by tools/export_model.py, and the offset of the "model" partition (see partitions.csv).
MODEL        ?= ethoric_int4.bin
MODEL_OFFSET ?= 0x110000
BAUD         ?= 921600

.PHONY: all build flash flash-model monitor config clean erase debug test

all: build

build:
	idf.py build

# Flash the firmware (fast). Repeat after every code change.
flash:
	idf.py -p $(PORT) flash

# Flash the model (~7-10 MB) into the "model" partition. Only needed when the model file changes.
flash-model:
	python -m esptool --chip esp32s3 -p $(PORT) -b $(BAUD) write_flash $(MODEL_OFFSET) $(MODEL)

monitor:
	idf.py -p $(PORT) monitor

config:
	idf.py menuconfig

clean:
	idf.py fullclean

erase:
	idf.py erase-flash -p $(PORT)

debug: build flash monitor

# Host-side tests (no hardware required).
test:
	$(MAKE) -C host test
