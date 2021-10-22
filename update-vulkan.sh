#!/bin/sh
curl -LO https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/main/xml/vk.xml
vulkan-zig-generator vk.xml src/vk.zig
