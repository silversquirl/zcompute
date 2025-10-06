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

    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    const test_exe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_exe.root_module.addImport("vk", vk);
    test_exe.linkLibC();

    const run_test = b.addRunArtifact(test_exe);
    b.step("test", "Run library tests").dependOn(&run_test.step);
}
