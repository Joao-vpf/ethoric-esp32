# Host-side build and tests (no ESP32 required).
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pthread -I../components/ethoric/include
BUILD    := build

all: $(BUILD)/ethoric_host $(BUILD)/test_sampler $(BUILD)/test_json $(BUILD)/test_dns

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/ethoric_host: host_main.cpp ../components/ethoric/include/ethoric.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ host_main.cpp

$(BUILD)/test_sampler: test_sampler.cpp ../components/ethoric/include/sampler.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ test_sampler.cpp

$(BUILD)/test_json: test_json.cpp ../components/ethoric/include/mini_json.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ test_json.cpp

$(BUILD)/test_dns: test_dns.cpp ../components/ethoric/include/dns_captive.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ test_dns.cpp

# Unit test for the sampler plus the full C++ vs. reference-decoder suite.
test: $(BUILD)/test_sampler $(BUILD)/test_json $(BUILD)/test_dns
	$(BUILD)/test_sampler
	$(BUILD)/test_json
	$(BUILD)/test_dns
	python3 ../tests/test_reference_decoder.py
	python3 ../tests/test_cpp_decoder.py
	python3 ../tests/test_tokenizer.py

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
