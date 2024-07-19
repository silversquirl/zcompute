const std = @import("std");

pub fn build(b: *std.Build) void {
    const vk = b.dependency("vulkan_zig", .{
        .registry = @as([]const u8, b.pathFromRoot("vk.xml")),
    }).module("vulkan-zig");

    _ = b.addModule("zcompute", .{
        .root_source_file = b.path("src/zcompute.zig"),
        .imports = &.{
            .{ .name = "vk", .module = vk },
        },
    });

    const test_exe = b.addTest(.{
        .root_source_file = b.path("src/test.zig"),
    });
    test_exe.root_module.addImport("vk", vk);
    test_exe.linkLibC();

    const run_test = b.addRunArtifact(test_exe);
    b.step("test", "Run library tests").dependOn(&run_test.step);
}
