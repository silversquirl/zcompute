#!/bin/sh
curl -LO https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/v1.3.240/xml/vk.xml
vulkan-zig-generator vk.xml src/vk.zig
