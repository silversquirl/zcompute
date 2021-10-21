#!/bin/sh
curl -LO https://raw.githubusercontent.com/KhronosGroup/Vulkan-Docs/master/xml/vk.xml
vulkan-zig-generator vk.xml vk.zig
