const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();
    example(b, target, mode, "collatz");
}

fn example(b: *std.build.Builder, target: std.build.Target, mode: std.builtin.Mode, name: []const u8) void {
    const exe = b.addExecutable(name, b.fmt("{s}/main.zig", .{name}));
    exe.addPackagePath("zcompute", "../src/zcompute.zig");
    exe.linkage = .dynamic;
    exe.linkLibC();
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step(b.fmt("run-{s}", .{name}), b.fmt("Run the {s} example", .{name}));
    run_step.dependOn(&run_cmd.step);
}
