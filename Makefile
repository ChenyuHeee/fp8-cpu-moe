CXX := g++
CPPFLAGS := -Isrc
CXXFLAGS := -O3 -std=c++17 -fPIC -fvisibility=hidden -Wall -Wextra -Wpedantic
LDFLAGS := -shared -pthread

BUILD_DIR := build
LIBRARY := $(BUILD_DIR)/libfp8_moe.so
OBJECT := $(BUILD_DIR)/fp8_moe.o

.PHONY: all clean test check-toolchain

all: check-toolchain $(LIBRARY)

check-toolchain:
	@major=`$(CXX) -dumpversion | cut -d. -f1`; \
	if [ "$$major" -lt 15 ]; then \
		echo "error: GCC 15+ is required; run: source /opt/rh/gcc-toolset-15/enable" >&2; \
		exit 1; \
	fi

$(BUILD_DIR):
	mkdir -p $@

$(OBJECT): src/fp8_moe.cpp src/fp8_moe.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(LIBRARY): $(OBJECT)
	$(CXX) $< $(LDFLAGS) -o $@

test: all
	python tests/test_parity.py

clean:
	rm -f $(OBJECT) $(LIBRARY)
