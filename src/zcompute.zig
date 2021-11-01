//! Simple and easy to use GPU compute library for Zig

const std = @import("std");
const builtin = @import("builtin");
const vk = @import("vk.zig");
const vk_allocator = @import("vk_allocator.zig");
const log = std.log.scoped(.zcompute);

pub const Context = struct {
    // Public fields:
    compute_dispatch_limits: [3]u32, // Maximum size of each compute dispatch dimension

    // Internal fields:
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
        if (builtin.mode != .Debug) {
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
        self.compute_dispatch_limits = props.limits.max_compute_work_group_count;

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

    fn alloc(self: *Context, size: u64, mem_type_bits: u32, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        if (self.alloc_count >= self.alloc_max) {
            return error.OutOfDeviceMemory;
        }
        const mem_type_idx = self.findMemoryType(size, mem_type_bits, flags) orelse {
            return error.UnsupportedAllocationFlags;
        };
        return self.vkd.allocateMemory(self.device, .{
            .allocation_size = size,
            .memory_type_index = mem_type_idx,
        }, &self.vk_alloc);
    }
    fn free(self: *Context, mem: vk.DeviceMemory) void {
        self.alloc_count -= 1;
        self.vkd.freeMemory(self.device, mem, &self.vk_alloc);
    }

    fn findMemoryType(self: Context, size: u64, mem_type_bits: u32, flags: vk.MemoryPropertyFlags) ?u32 {
        // TODO: if host_visible, prioritize host_cached
        const mems = self.vki.getPhysicalDeviceMemoryProperties(self.phys_device);
        const mem_types = std.bit_set.IntegerBitSet(32){ .mask = mem_type_bits };
        var it = mem_types.iterator(.{});
        while (it.next()) |mem_type_idx| {
            const mem_type = mems.memory_types[0..mems.memory_type_count][mem_type_idx];
            if (mem_type.property_flags.contains(flags) and
                size <= mems.memory_heaps[mem_type.heap_index].size)
            {
                return @intCast(u32, mem_type_idx);
            }
        }
        return null;
    }
};

pub fn Shader(comptime PushConstants: type, comptime binding_points: []const ShaderBinding) type {
    comptime var layout_bindings: [binding_points.len]vk.DescriptorSetLayoutBinding = undefined;
    comptime var desc_template_entries: [binding_points.len]vk.DescriptorUpdateTemplateEntry = undefined;
    comptime var type_counts = std.EnumArray(ShaderBindingType, u32).initDefault(0, .{});
    for (binding_points) |bind, i| {
        if (!isBufferWrapper(bind[3])) {
            @compileError("Shader bindings must use buffer types");
        }

        layout_bindings[i] = .{
            .binding = bind[1],
            .descriptor_type = bind[2].toVk(),
            .descriptor_count = 1,
            .stage_flags = .{ .compute_bit = true },
            .p_immutable_samplers = null,
        };

        // TODO: merge consecutive bindings
        desc_template_entries[i] = .{
            .dst_binding = bind[1],
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = bind[2].toVk(),
            .offset = i * @sizeOf(vk.DescriptorBufferInfo),
            .stride = 0, // TODO
        };

        type_counts.getPtr(bind[2]).* += 1;
    }

    const pool_sizes = blk: {
        var pool_sizes: [type_counts.values.len]vk.DescriptorPoolSize = undefined;
        var i = 0;
        var it = type_counts.iterator();
        while (it.next()) |entry| {
            if (entry.value.* > 0) {
                pool_sizes[i] = .{
                    .type = entry.key.toVk(),
                    // Multiply by 2 because we need two descriptor sets
                    .descriptor_count = 2 * entry.value.*,
                };
                i += 1;
            }
        }

        break :blk pool_sizes[0..i].*;
    };

    return struct {
        ctx: *Context,

        desc_layout: vk.DescriptorSetLayout,
        pipeline_layout: vk.PipelineLayout,
        pipeline: vk.Pipeline,
        cmd_pool: vk.CommandPool,
        desc_pool: vk.DescriptorPool,
        desc_template: vk.DescriptorUpdateTemplate,

        fence: vk.Fence,
        idx: u1 = 0,
        cmd_bufs: [2]vk.CommandBuffer,
        desc_sets: [2]vk.DescriptorSet,

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
                .binding_count = @intCast(u32, layout_bindings.len),
                .p_bindings = &layout_bindings,
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, desc_layout, &ctx.vk_alloc);

            const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, .{
                .flags = .{},
                .set_layout_count = 1,
                .p_set_layouts = &[_]vk.DescriptorSetLayout{desc_layout},
                .push_constant_range_count = @boolToInt(@sizeOf(PushConstants) > 0),
                .p_push_constant_ranges = &[1]vk.PushConstantRange{.{
                    .stage_flags = .{ .compute_bit = true },
                    .offset = 0,
                    .size = @sizeOf(PushConstants),
                }},
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, &ctx.vk_alloc);

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
            errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline[0], &ctx.vk_alloc);

            const cmd_pool = try ctx.vkd.createCommandPool(ctx.device, .{
                .flags = .{
                    .transient_bit = true,
                    .reset_command_buffer_bit = true,
                },
                .queue_family_index = ctx.queue_family,
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyCommandPool(ctx.device, cmd_pool, &ctx.vk_alloc);

            const desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, .{
                .flags = .{},
                .max_sets = 2,
                .pool_size_count = pool_sizes.len,
                .p_pool_sizes = &pool_sizes,
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, &ctx.vk_alloc);

            const desc_template = try ctx.vkd.createDescriptorUpdateTemplate(ctx.device, .{
                .flags = .{},
                .descriptor_update_entry_count = desc_template_entries.len,
                .p_descriptor_update_entries = &desc_template_entries,
                .template_type = .descriptor_set,
                .descriptor_set_layout = desc_layout,
                // These fields are ignored in Vulkan 1.1, but we have them so we might as well provide them
                .pipeline_bind_point = .compute,
                .pipeline_layout = pipeline_layout,
                .set = undefined, // idk what this one is
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyDescriptorUpdateTemplate(ctx.device, desc_template, &ctx.vk_alloc);

            const fence = try ctx.vkd.createFence(ctx.device, .{
                .flags = .{ .signaled_bit = true },
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyFence(ctx.device, fence, &ctx.vk_alloc);

            var cmd_bufs: [2]vk.CommandBuffer = undefined;
            try ctx.vkd.allocateCommandBuffers(ctx.device, .{
                .command_pool = cmd_pool,
                .level = .primary,
                .command_buffer_count = 2,
            }, &cmd_bufs);

            var desc_sets: [2]vk.DescriptorSet = undefined;
            try ctx.vkd.allocateDescriptorSets(ctx.device, .{
                .descriptor_pool = desc_pool,
                .descriptor_set_count = 2,
                .p_set_layouts = &[2]vk.DescriptorSetLayout{
                    desc_layout,
                    desc_layout,
                },
            }, &desc_sets);

            return Self{
                .ctx = ctx,

                .desc_layout = desc_layout,
                .pipeline_layout = pipeline_layout,
                .pipeline = pipeline[0],
                .cmd_pool = cmd_pool,
                .desc_pool = desc_pool,
                .desc_template = desc_template,

                .fence = fence,
                .cmd_bufs = cmd_bufs,
                .desc_sets = desc_sets,
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
            self.ctx.vkd.destroyFence(self.ctx.device, self.fence, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyDescriptorUpdateTemplate(self.ctx.device, self.desc_template, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyDescriptorPool(self.ctx.device, self.desc_pool, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyCommandPool(self.ctx.device, self.cmd_pool, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyPipelineLayout(self.ctx.device, self.pipeline_layout, &self.ctx.vk_alloc);
            self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.device, self.desc_layout, &self.ctx.vk_alloc);
        }

        /// Waits indefinitely for the current execution to complete
        pub fn wait(self: Self) !void {
            while (!try self.waitTimeout(~@as(u64, 0))) {}
        }

        /// Waits for the current execution to complete, returning true on completion.
        /// If the timeout is reached, returns false.
        pub fn waitTimeout(self: Self, timeout: u64) !bool {
            const res = try self.ctx.vkd.waitForFences(
                self.ctx.device,
                1,
                &[1]vk.Fence{self.fence},
                vk.TRUE,
                timeout,
            );
            return res == .success;
        }

        /// Builds a new execution of the shader, waits for any previous execution to complete, then submits the execution.
        /// If you know any previous execution has completed prior to this exec call, pass 0 as the wait_timeout.
        pub fn exec(
            self: *Self,
            wait_timeout: ?u64,
            dispatch: ComputeDispatch,
            push_constants: PushConstants,
            bindings: Bindings,
        ) !void {
            // Update descriptor set
            const desc_set = self.desc_sets[self.idx];

            var descriptors: [binding_points.len]vk.DescriptorBufferInfo = undefined;
            inline for (std.meta.fields(Bindings)) |field, i| {
                descriptors[i] = .{
                    .buffer = @field(bindings, field.name).buf,
                    .offset = 0,
                    .range = vk.WHOLE_SIZE,
                };
            }
            self.ctx.vkd.updateDescriptorSetWithTemplate(
                self.ctx.device,
                desc_set,
                self.desc_template,
                @ptrCast(*const c_void, &descriptors),
            );

            // Record command buffer
            const cmd_buf = self.cmd_bufs[self.idx];
            try self.ctx.vkd.beginCommandBuffer(cmd_buf, .{
                .flags = .{ .one_time_submit_bit = true },
                .p_inheritance_info = null,
            });

            // Bind pipeline
            self.ctx.vkd.cmdBindPipeline(cmd_buf, .compute, self.pipeline);
            // Bind descriptor set
            self.ctx.vkd.cmdBindDescriptorSets(cmd_buf, // Why does this function not take a struct??
                .compute, self.pipeline_layout, // pipeline info
                0, 1, &[1]vk.DescriptorSet{desc_set}, // descriptor sets
                0, undefined // dynamic offsets
            );
            if (@sizeOf(PushConstants) > 0) {
                // Send push constants
                self.ctx.vkd.cmdPushConstants(
                    cmd_buf,
                    self.pipeline_layout,
                    .{ .compute_bit = true },
                    0,
                    @sizeOf(PushConstants),
                    @ptrCast(*const c_void, &push_constants),
                );
            }

            // Dispatch compute
            self.ctx.vkd.cmdDispatchBase(
                cmd_buf,
                dispatch.baseX,
                dispatch.baseY,
                dispatch.baseZ,
                dispatch.x,
                dispatch.y,
                dispatch.z,
            );

            try self.ctx.vkd.endCommandBuffer(cmd_buf);

            // Submit execution
            if (wait_timeout) |timeout| {
                if (!try self.waitTimeout(timeout)) {
                    return error.Timeout;
                }
            } else {
                try self.wait();
            }
            try self.ctx.vkd.resetFences(self.ctx.device, 1, &[1]vk.Fence{self.fence});
            try self.ctx.vkd.queueSubmit(self.ctx.queue, 1, &[1]vk.SubmitInfo{.{
                .command_buffer_count = 1,
                .p_command_buffers = &[1]vk.CommandBuffer{cmd_buf},

                .wait_semaphore_count = 0,
                .p_wait_semaphores = undefined,
                .p_wait_dst_stage_mask = undefined,
                .signal_semaphore_count = 0,
                .p_signal_semaphores = undefined,
            }}, self.fence);

            // Swap buffers
            self.idx ^= 1;
        }

        pub const Bindings = blk: {
            var fields: [binding_points.len]std.builtin.TypeInfo.StructField = undefined;
            for (binding_points) |bind, i| {
                fields[i] = .{
                    .name = bind[0],
                    .field_type = bind[3],
                    .default_value = null,
                    .is_comptime = false,
                    .alignment = @alignOf(bind[3]),
                };
            }
            break :blk @Type(.{ .Struct = .{
                .layout = .Auto,
                .fields = &fields,
                .decls = &.{},
                .is_tuple = false,
            } });
        };
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

    fn toVk(t: ShaderBindingType) vk.DescriptorType {
        return switch (t) {
            .uniform => .uniform_buffer,
            .storage => .storage_buffer,
        };
    }
};
pub const ComputeDispatch = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
    baseX: u32 = 0,
    baseY: u32 = 0,
    baseZ: u32 = 0,
};

pub fn Buffer(comptime T: type) type {
    return struct {
        ctx: *Context,
        buf: vk.Buffer,
        mem: vk.DeviceMemory,
        off: u64,
        len: u64,

        const Self = @This();
        pub const is_zcompute_buffer_wrapper = void;

        pub fn init(ctx: *Context, len: u64, flags: BufferInitFlags) !Self {
            const buf = try ctx.vkd.createBuffer(ctx.device, .{
                .flags = .{},
                .size = len * @sizeOf(T),
                .usage = .{
                    .uniform_buffer_bit = flags.uniform,
                    .storage_buffer_bit = flags.storage,
                },
                .sharing_mode = .exclusive,
                .queue_family_index_count = 1,
                .p_queue_family_indices = &[_]u32{
                    ctx.queue_family,
                },
            }, &ctx.vk_alloc);
            errdefer ctx.vkd.destroyBuffer(ctx.device, buf, &ctx.vk_alloc);

            const reqs = ctx.vkd.getBufferMemoryRequirements(ctx.device, buf);
            const mem = try ctx.alloc(reqs.size, reqs.memory_type_bits, .{
                .host_coherent_bit = flags.coherent,
                .host_visible_bit = flags.map,
            });
            errdefer ctx.free(mem);
            try ctx.vkd.bindBufferMemory(ctx.device, buf, mem, 0);

            return Self{
                .ctx = ctx,
                .buf = buf,
                .mem = mem,
                .off = 0,
                .len = len,
            };
        }

        pub fn deinit(self: Self) void {
            self.ctx.free(self.mem);
            self.ctx.vkd.destroyBuffer(self.ctx.device, self.buf, &self.ctx.vk_alloc);
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
fn isBufferWrapper(comptime T: type) bool {
    return @typeInfo(T) == .Struct and
        @hasField(T, "buf") and
        std.meta.fieldInfo(T, .buf).field_type == vk.Buffer and
        @hasDecl(T, "is_zcompute_buffer_wrapper");
}

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
    .AllocateCommandBuffers,
    .AllocateDescriptorSets,
    .AllocateMemory,
    .BeginCommandBuffer,
    .BindBufferMemory,
    .CmdBindDescriptorSets,
    .CmdBindPipeline,
    .CmdDispatchBase,
    .CmdPushConstants,
    .CreateBuffer,
    .CreateCommandPool,
    .CreateComputePipelines,
    .CreateDescriptorPool,
    .CreateDescriptorSetLayout,
    .CreateDescriptorUpdateTemplate,
    .CreateFence,
    .CreatePipelineLayout,
    .CreateShaderModule,
    .DestroyBuffer,
    .DestroyCommandPool,
    .DestroyDescriptorPool,
    .DestroyDescriptorSetLayout,
    .DestroyDescriptorUpdateTemplate,
    .DestroyDevice,
    .DestroyFence,
    .DestroyPipeline,
    .DestroyPipelineLayout,
    .DestroyShaderModule,
    .EndCommandBuffer,
    .FreeMemory,
    .GetBufferMemoryRequirements,
    .GetDeviceQueue,
    .MapMemory,
    .QueueSubmit,
    .ResetFences,
    .UnmapMemory,
    .UpdateDescriptorSetWithTemplate,
    .WaitForFences,
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

        const lib_name = switch (builtin.os.tag) {
            .windows => "vulkan-1.dll",
            else => "libvulkan.so.1",
            .macos => @compileError("Unsupported platform: " ++ @tagName(builtin.os)),
        };
        if (!builtin.link_libc) {
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
