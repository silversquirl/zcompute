const std = @import("std");
const zc = @import("zcompute");

const histogram_len = 800;
const max_n = 20_000_000;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{
        // If we let GPA record stack traces, it can cause segfaults from within Vulkan
        .stack_trace_frames = 0,
    }){};
    errdefer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a new zcompute context
    // This sets up a link to the GPU, and is required for all usage of the library
    var ctx = try zc.Context.init(allocator, .{});
    defer ctx.deinit();

    // Create a compute shader with the right interface
    var shad = try zc.Shader(&.{
        zc.pushConstant("histogram_len", 0, u32), // This is a push constant
        zc.storageBuffer("histogram", 0, zc.Buffer(u32)), // This is a storage buffer
    }).initBytes(allocator, &ctx, @embedFile("collatz.spv"));
    defer shad.deinit();

    // Allocate a buffer for histogram data
    const histogram_buf = try zc.Buffer(u32).init(&ctx, histogram_len, .{
        .map = true,
        .storage = true,
    });
    defer histogram_buf.deinit();

    // Execute the compute shader
    // We do this in multiple batches because the number of workgroups is too high to do all at once
    var progress = std.Progress{};
    {
        const max_batch = ctx.compute_dispatch_limits[0]; // Maximum size of one batch
        std.debug.print("Computing collatz stopping time histogram for values up to {}, at batch size {}\n", .{ max_n, max_batch });

        const node = progress.start("Executing batches...", (max_n - 1) / max_batch + 1);
        defer node.end();

        var n: u32 = 0;
        while (n < max_n) : (n += max_batch) {
            const batch = @min(max_n - n, max_batch);
            try shad.exec(null, .{
                // This struct defines the dimensions of the compute dispatch
                .x = batch,
                .baseX = n, // Start the current batch where the previous batch finished
            }, .{
                // This is where we pass parameters, according to the format we defined when creating the shader type
                .histogram_len = histogram_len,
                .histogram = histogram_buf,
            });
            node.completeOne();
        }
    }
    std.debug.print("Done in {}\n", .{std.fmt.fmtDuration(progress.timer.?.read())});

    // Wait for the last shader to finish
    // Shader execution happens asynchronously, so if we want to read back the data we need to synchronize
    // We only need to do this at the end, because exec will wait for the previous execution
    try shad.wait();

    // Read back the computed data and output it as CSV
    {
        const histogram_data = try histogram_buf.map();
        defer histogram_buf.unmap();
        const out = std.io.getStdOut().writer();
        for (histogram_data) |value, i| {
            if (i > 0) try out.writeByte(',');
            try out.print("{}", .{value});
        }
        try out.writeByte('\n');
    }
}
