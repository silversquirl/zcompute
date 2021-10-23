//! Simple and easy to use GPU compute library for Zig

const std = @import("std");
const vk = @import("vk.zig");
const vk_allocator = @import("vk_allocator.zig");
const log = std.log.scoped(.zcompute);

pub const Context = struct {
    allocator: *std.mem.Allocator,
    vk_alloc: vk.AllocationCallbacks,

    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    phys_device: vk.PhysicalDevice,
    device: vk.Device,

    queue_family: u32,
    queue: vk.Queue,

    alloc_count: u32 = 0,
    alloc_max: u32,

    /// WARNING: Using a GPA with nonzero stack_trace_frames may cause random segmentation faults
    pub fn init(allocator: *std.mem.Allocator) !Context {
        var self: Context = undefined;
        self.allocator = allocator;
        self.vk_alloc = vk_allocator.wrap(allocator);

        try loader.ref();
        errdefer loader.deref();
        self.vkb = try BaseDispatch.load(loader.getProcAddress);

        try self.initInstance(allocator);
        errdefer self.vki.destroyInstance(self.instance, &self.vk_alloc);

        try self.initDevice(allocator);
        errdefer self.vkd.destroyDevice(self.device, &self.vk_alloc);

        return self;
    }

    pub fn deinit(self: Context) void {
        self.vkd.destroyDevice(self.device, &self.vk_alloc);
        self.vki.destroyInstance(self.instance, &self.vk_alloc);
        loader.deref();
    }

    fn initInstance(self: *Context, allocator: *std.mem.Allocator) !void {
        const app_name = std.meta.globalOption("zcompute_app_name", [*:0]const u8);
        const app_version = std.meta.globalOption("zcompute_app_version", u32) orelse 0;
        const layers = try self.instanceLayers(allocator);
        defer allocator.free(layers);

        self.instance = try self.vkb.createInstance(.{
            .flags = .{},
            .p_application_info = &.{
                .p_application_name = app_name,
                .application_version = app_version,
                .p_engine_name = "zcompute",
                .engine_version = 00_01_00,
                .api_version = vk.makeApiVersion(0, 1, 1, 0),
            },
            .enabled_layer_count = @intCast(u32, layers.len),
            .pp_enabled_layer_names = layers.ptr,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = undefined,
        }, &self.vk_alloc);
        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn instanceLayers(self: Context, allocator: *std.mem.Allocator) ![][*:0]const u8 {
        if (std.builtin.mode != .Debug) {
            return &.{};
        }

        var wanted_layers = [_][:0]const u8{
            "VK_LAYER_KHRONOS_validation",
        };

        var n_supported_layers: u32 = undefined;
        _ = try self.vkb.enumerateInstanceLayerProperties(&n_supported_layers, null);
        const supported_layers = try allocator.alloc(vk.LayerProperties, n_supported_layers);
        defer allocator.free(supported_layers);
        _ = try self.vkb.enumerateInstanceLayerProperties(&n_supported_layers, supported_layers.ptr);

        var n_layers: usize = 0;
        var layers: [wanted_layers.len][*:0]const u8 = undefined;
        for (wanted_layers) |wanted| {
            for (supported_layers[0..n_supported_layers]) |supported| {
                if (std.mem.eql(u8, wanted, std.mem.sliceTo(&supported.layer_name, 0))) {
                    layers[n_layers] = wanted.ptr;
                    n_layers += 1;
                    break;
                }
            } else {
                log.warn("Skipping validation layer {s}", .{wanted});
            }
        }

        return allocator.dupe([*:0]const u8, layers[0..n_layers]);
    }

    fn initDevice(self: *Context, allocator: *std.mem.Allocator) !void {
        // Find best physical device
        var n_devices: u32 = undefined;
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &n_devices, null);
        const devices = try allocator.alloc(vk.PhysicalDevice, n_devices);
        defer allocator.free(devices);
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &n_devices, devices.ptr);

        self.phys_device = for (devices[0..n_devices]) |dev| {
            var n_queues: u32 = undefined;
            self.vki.getPhysicalDeviceQueueFamilyProperties(dev, &n_queues, null);
            const queues = try allocator.alloc(vk.QueueFamilyProperties, n_queues);
            defer allocator.free(queues);
            self.vki.getPhysicalDeviceQueueFamilyProperties(dev, &n_queues, queues.ptr);

            self.queue_family = for (queues[0..n_queues]) |queue, i| {
                if (queue.queue_flags.compute_bit) {
                    break @intCast(u32, i);
                }
            } else {
                continue;
            };

            break dev;
        } else {
            return error.NoSuitableDevice;
        };

        const props = self.vki.getPhysicalDeviceProperties(self.phys_device);
        self.alloc_max = props.limits.max_memory_allocation_count;

        // Create logical device
        const queue_infos = [_]vk.DeviceQueueCreateInfo{
            .{
                .flags = .{},
                .queue_family_index = self.queue_family,
                .queue_count = 1,
                .p_queue_priorities = &[1]f32{1.0},
            },
        };

        self.device = try self.vki.createDevice(self.phys_device, .{
            .flags = .{},
            .queue_create_info_count = queue_infos.len,
            .p_queue_create_infos = &queue_infos,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = undefined,
            .p_enabled_features = null,
        }, &self.vk_alloc);
        self.vkd = try DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr);
        errdefer self.vkd.destroyDevice(self.device, &self.vk_alloc);

        self.queue = self.vkd.getDeviceQueue(self.device, self.queue_family, 0);
    }

    pub fn alloc(self: *Context, size: u64, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        if (self.alloc_count >= self.alloc_max) {
            return error.OutOfDeviceMemory;
        }
        const mem_type_idx = self.findMemoryType(size, flags) orelse {
            return error.UnsupportedAllocationFlags;
        };
        return self.vkd.allocateMemory(self.device, .{
            .allocation_size = size,
            .memory_type_index = mem_type_idx,
        }, &self.vk_alloc);
    }
    pub fn free(self: *Context, mem: vk.DeviceMemory) void {
        self.vkd.freeMemory(self.device, mem, &self.vk_alloc);
    }

    fn findMemoryType(self: Context, size: u64, flags: vk.MemoryPropertyFlags) ?u32 {
        // TODO: if host_visible, prioritize host_cached
        const mems = self.vki.getPhysicalDeviceMemoryProperties(self.phys_device);
        for (mems.memory_types[0..mems.memory_type_count]) |mem_type, mem_type_idx| {
            if (mem_type.property_flags.contains(flags) and
                size < mems.memory_heaps[mem_type.heap_index].size)
            {
                return @intCast(u32, mem_type_idx);
            }
        }
        return null;
    }
};

pub fn Shader(comptime bindings: []const ShaderBinding) type {
    const vk_bindings = blk: {
        var vk_bindings: [bindings.len]vk.DescriptorSetLayoutBinding = undefined;
        for (bindings) |bind, i| {
            vk_bindings[i] = .{
                .binding = bind[1],
                .descriptor_type = switch (bind[2]) {
                    .uniform => .uniform_buffer,
                    .storage => .storage_buffer,
                },
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
                .p_immutable_samplers = null,
            };
        }
        break :blk vk_bindings;
    };

    return struct {
        ctx: *Context,
        pipeline: vk.Pipeline,

        const Self = @This();

        // Creates a shader from an array of native-endian u32
        pub fn init(ctx: *Context, code: []const u32) !Self {
            const module = try ctx.vkd.createShaderModule(ctx.device, .{
                .flags = .{},
                .code_size = 4 * code.len,
                .p_code = code.ptr,
            }, &ctx.vk_alloc);
            defer ctx.vkd.destroyShaderModule(ctx.device, module, &ctx.vk_alloc);

            // TODO: cache descriptor sets?
            const desc_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, .{
                .flags = .{},
                .binding_count = @intCast(u32, vk_bindings.len),
                .p_bindings = &vk_bindings,
            }, &ctx.vk_alloc);
            defer ctx.vkd.destroyDescriptorSetLayout(ctx.device, desc_layout, &ctx.vk_alloc);

            const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, .{
                .flags = .{},
                .set_layout_count = 1,
                .p_set_layouts = &[_]vk.DescriptorSetLayout{desc_layout},
                .push_constant_range_count = 0,
                .p_push_constant_ranges = undefined,
            }, &ctx.vk_alloc);
            defer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, &ctx.vk_alloc);

            var pipeline: [1]vk.Pipeline = undefined;
            _ = try ctx.vkd.createComputePipelines(
                ctx.device,
                .null_handle, // TODO: pipeline caching?
                1,
                &[_]vk.ComputePipelineCreateInfo{.{
                    .flags = .{},
                    .stage = .{
                        .flags = .{},
                        .stage = .{ .compute_bit = true },
                        .module = module,
                        .p_name = "main",
                        .p_specialization_info = null,
                    },
                    .layout = pipeline_layout,
                    .base_pipeline_handle = .null_handle,
                    .base_pipeline_index = 0,
                }},
                &ctx.vk_alloc,
                &pipeline,
            );
            errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline, &ctx.vk_alloc);

            return Self{
                .ctx = ctx,
                .pipeline = pipeline[0],
            };
        }

        // Creates a shader from a array of bytes
        pub fn initBytes(ctx: *Context, code: []const u8) !Self {
            if (code.len & 3 != 0 or code.len == 0) {
                return error.InvalidShader;
            }

            // Detect endianness
            const magic = std.mem.readIntSliceLittle(u32, code);
            const spirv_magic = 0x07230203;
            const endian: std.builtin.Endian = switch (magic) {
                spirv_magic => .Little,
                @byteSwap(u32, spirv_magic) => .Big,
                else => return error.InvalidShader,
            };

            // Read SPIR-V
            const code32 = try ctx.allocator.alloc(u32, @divExact(code.len, 4));
            defer ctx.allocator.free(code32);
            for (code32) |*v, i| {
                v.* = std.mem.readIntSlice(u32, code[i * 4 ..], endian);
            }

            // Init shader
            return init(ctx, code32);
        }

        pub fn deinit(self: Self) void {
            self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline, &self.ctx.vk_alloc);
        }
    };
}
pub const ShaderBinding = std.meta.Tuple(&.{
    []const u8, // Field name
    u32, // Binding index
    ShaderBindingType, // Binding type
    type, // Data type
});
pub const ShaderBindingType = enum {
    uniform,
    storage,
};

pub fn Buffer(comptime T: type) type {
    return struct {
        ctx: *Context,
        buf: vk.Buffer,
        mem: vk.DeviceMemory,
        off: u64,
        len: u64,
        owned: bool,

        const Self = @This();

        pub fn init(ctx: *Context, len: u64, flags: BufferInitFlags) !Self {
            const mem = try ctx.alloc(len * @sizeOf(T), .{
                .host_coherent_bit = flags.coherent,
                .host_visible_bit = flags.map,
            });
            var self = try initMem(ctx, mem, 0, len, flags, true, true);
            return self;
        }

        pub fn initMem(
            ctx: *Context,
            mem: vk.DeviceMemory,
            off: u64,
            len: u64,
            flags: BufferInitFlags,
            own_memory: bool,
            exclusive: bool,
        ) !Self {
            std.debug.assert(!own_memory or exclusive); // If owned, must also be exclusive

            const buf = try ctx.vkd.createBuffer(ctx.device, .{
                .flags = .{},
                .size = len * @sizeOf(T),
                .usage = .{
                    .uniform_buffer_bit = flags.uniform,
                    .storage_buffer_bit = flags.storage,
                },
                .sharing_mode = if (exclusive) .exclusive else .concurrent,
                .queue_family_index_count = 1,
                .p_queue_family_indices = &[_]u32{
                    ctx.queue_family,
                },
            }, &ctx.vk_alloc);

            return Self{
                .ctx = ctx,
                .buf = buf,
                .mem = mem,
                .off = off,
                .len = len,
                .owned = own_memory,
            };
        }

        pub fn deinit(self: Self) void {
            self.ctx.vkd.destroyBuffer(self.ctx.device, self.buf, &self.ctx.vk_alloc);
            if (self.owned) {
                self.ctx.free(self.mem);
            }
        }

        const min_map_align = 64; // Spec requires min_memory_map_alignment limit to be at least 64
        pub fn map(self: Self) ![]align(min_map_align) T {
            const ptr = try self.ctx.vkd.mapMemory(
                self.ctx.device,
                self.mem,
                self.off,
                self.len * @sizeOf(T),
                .{},
            );
            const ptr_aligned = @alignCast(min_map_align, ptr);
            return @ptrCast([*]align(min_map_align) T, ptr_aligned)[0..self.len];
        }
        pub fn unmap(self: Self) void {
            self.ctx.vkd.unmapMemory(self.ctx.device, self.mem);
        }
    };
}
pub const BufferInitFlags = packed struct {
    coherent: bool = false,
    map: bool = false,

    uniform: bool = false,
    storage: bool = false,
};

const BaseDispatch = vk.BaseWrapper(.{
    .CreateInstance,
    .EnumerateInstanceLayerProperties,
    .GetInstanceProcAddr,
});

const InstanceDispatch = vk.InstanceWrapper(.{
    .CreateDevice,
    .DestroyInstance,
    .EnumerateDeviceExtensionProperties,
    .EnumeratePhysicalDevices,
    .GetDeviceProcAddr,
    .GetPhysicalDeviceMemoryProperties,
    .GetPhysicalDeviceProperties,
    .GetPhysicalDeviceQueueFamilyProperties,
});

const DeviceDispatch = vk.DeviceWrapper(.{
    .AllocateMemory,
    .CreateBuffer,
    .CreateComputePipelines,
    .CreateDescriptorSetLayout,
    .CreatePipelineLayout,
    .CreateShaderModule,
    .DestroyBuffer,
    .DestroyDescriptorSetLayout,
    .DestroyDevice,
    .DestroyPipeline,
    .DestroyPipelineLayout,
    .DestroyShaderModule,
    .FreeMemory,
    .GetDeviceQueue,
    .MapMemory,
    .UnmapMemory,
});

// Simple loader for base Vulkan functions
threadlocal var loader = Loader{};
const Loader = struct {
    ref_count: usize = 0,
    lib: ?std.DynLib = null,
    getProcAddress: vk.PfnGetInstanceProcAddr = undefined,

    fn ref(self: *Loader) !void {
        if (self.lib != null) {
            self.ref_count += 1;
            return;
        }

        const lib_name = switch (std.builtin.os.tag) {
            .windows => "vulkan-1.dll",
            else => "libvulkan.so.1",
            .macos => @compileError("Unsupported platform: " ++ @tagName(std.builtin.os)),
        };
        if (!std.builtin.link_libc) {
            @compileError("zcompute requires libc to be linked");
        }

        self.lib = std.DynLib.open(lib_name) catch |err| {
            log.err("Could not load vulkan library '{s}': {s}", .{ lib_name, @errorName(err) });
            return err;
        };
        errdefer self.lib.?.close();

        self.getProcAddress = self.lib.?.lookup(
            vk.PfnGetInstanceProcAddr,
            "vkGetInstanceProcAddr",
        ) orelse {
            log.err("Vulkan loader does not export vkGetInstanceProcAddr", .{});
            return error.MissingSymbol;
        };
    }

    fn deref(self: *Loader) void {
        if (self.ref_count > 0) {
            self.ref_count -= 1;
            return;
        }

        self.lib.?.close();
        self.lib = null;
        self.getProcAddress = undefined;
    }
};
