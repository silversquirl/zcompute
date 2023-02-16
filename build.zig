const std = @import("std");

pub fn build(b: *std.Build) void {
    b.addModule(.{
        .name = "zcompute",
        .source_file = .{ .path = "src/zcompute.zig" },
    });

    const test_step = b.addTest(.{
        .root_source_file = .{ .path = "src/test.zig" },
    });
    test_step.linkLibC();
    b.step("test", "Run library tests").dependOn(&test_step.step);
}
