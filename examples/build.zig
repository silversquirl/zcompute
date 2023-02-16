const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    example(b, target, optimize, "collatz");
}

fn example(b: *std.build.Builder, target: std.zig.CrossTarget, optimize: std.builtin.Mode, name: []const u8) void {
    const exe = b.addExecutable(.{
        .name = name,
        .root_source_file = .{ .path = b.fmt("{s}/main.zig", .{name}) },
        .target = target,
        .optimize = optimize,
    });
    exe.addAnonymousModule("zcompute", .{
        .source_file = .{ .path = "../src/zcompute.zig" },
    });
    exe.linkage = .dynamic;
    exe.linkLibC();
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step(b.fmt("run-{s}", .{name}), b.fmt("Run the {s} example", .{name}));
    run_step.dependOn(&run_cmd.step);
}
