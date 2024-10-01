//! Simple and easy to use GPU compute library for Zig

const std = @import("std");
const builtin = @import("builtin");
pub const vk = @import("vk");
const log = std.log.scoped(.zcompute);
const root = @import("root");

pub const VkDeviceFeatures = vk.PhysicalDeviceFeatures;

pub const Context = struct {
    // Public fields
    compute_dispatch_limits: [3]u32, // Maximum size of each compute dispatch dimension

    // Public-ish fields - usable, but be careful
    device_properties: vk.PhysicalDeviceProperties,

    // Internal fields - only use if you know exactly what you're doing
    vkb: BaseDispatch,
    vki: InstanceDispatch,
    vkd: DeviceDispatch,

    instance: vk.Instance,
    phys_device: vk.PhysicalDevice,
    device: vk.Device,

    queue_family: u32,
    queue: vk.Queue,

    alloc_count: u32,
    alloc_max: u32,

    const InitOptions = struct {
        vulkan_device_extensions: [][*:0]const u8 = &.{},
        vulkan_device_features: VkDeviceFeatures = .{},
    };

    /// WARNING: Using a GPA with nonzero stack_trace_frames may cause random segmentation faults
    /// NOTE: The allocator is only used for temporary allocations during setup.
    ///       Once init returns, all allocations created with it will be freed.
    ///       Vulkan allocations are done through libc's malloc implementation.
    pub fn init(allocator: std.mem.Allocator, opts: InitOptions) !Context {
        var self: Context = undefined;
        self.alloc_count = 0;

        try loader.ref();
        errdefer loader.deref();
        self.vkb = try BaseDispatch.load(loader.getProcAddress);

        try self.initInstance(allocator);
        errdefer self.vki.destroyInstance(self.instance, null);

        try self.initDevice(allocator, opts);
        errdefer self.vkd.destroyDevice(self.device, null);

        return self;
    }

    pub fn deinit(self: Context) void {
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
        loader.deref();
    }

    fn initInstance(self: *Context, allocator: std.mem.Allocator) !void {
        const app_name: ?[*:0]const u8 = if (@hasDecl(root, "zcompute_app_name")) root.zcompute_app_name else null;
        const app_version: u32 = if (@hasDecl(root, "zcompute_app_version")) root.zcompute_app_version else 0;
        const layers = try self.instanceLayers(allocator);
        defer allocator.free(layers);

        self.instance = try self.vkb.createInstance(&vk.InstanceCreateInfo{
            .flags = .{},
            .p_application_info = &.{
                .p_application_name = app_name,
                .application_version = app_version,
                .p_engine_name = "zcompute",
                .engine_version = 0 << 20 | 1 << 10 | 0,
                .api_version = vk.makeApiVersion(0, 1, 1, 0),
            },
            .enabled_layer_count = @intCast(layers.len),
            .pp_enabled_layer_names = layers.ptr,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = undefined,
        }, null);
        self.vki = try InstanceDispatch.load(self.instance, self.vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn instanceLayers(self: Context, allocator: std.mem.Allocator) ![][*:0]const u8 {
        if (builtin.mode != .Debug) {
            return &.{};
        }

        const wanted_layers: []const [:0]const u8 = if (@hasDecl(root, "zcompute_debug_layers"))
            root.zcompute_debug_layers
        else
            &.{"VK_LAYER_KHRONOS_validation"};

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
                log.warn("Skipping missing validation layer {s}", .{wanted});
            }
        }

        return allocator.dupe([*:0]const u8, layers[0..n_layers]);
    }

    fn initDevice(self: *Context, allocator: std.mem.Allocator, opts: InitOptions) !void {
        // Find best physical device
        var n_devices: u32 = undefined;
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &n_devices, null);
        const devices = try allocator.alloc(vk.PhysicalDevice, n_devices);
        defer allocator.free(devices);
        _ = try self.vki.enumeratePhysicalDevices(self.instance, &n_devices, devices.ptr);

        var ext_set = std.StringHashMap(void).init(allocator);
        for (opts.vulkan_device_extensions) |ext| {
            try ext_set.put(std.mem.span(ext), {});
        }

        self.phys_device = for (devices[0..n_devices]) |dev| {
            // Check device features
            const features = self.vki.getPhysicalDeviceFeatures(dev);
            const ok = inline for (comptime std.meta.fieldNames(VkDeviceFeatures)) |field| {
                const want = @field(opts.vulkan_device_features, field) != 0;
                const have = @field(features, field) != 0;
                if (want and !have) {
                    break false;
                }
            } else true;
            if (!ok) continue;

            // Check device extensions
            var n_exts: u32 = undefined;
            _ = try self.vki.enumerateDeviceExtensionProperties(dev, null, &n_exts, null);
            const exts = try allocator.alloc(vk.ExtensionProperties, n_exts);
            defer allocator.free(exts);
            _ = try self.vki.enumerateDeviceExtensionProperties(dev, null, &n_exts, exts.ptr);
            var n_matched: u32 = 0;
            for (exts[0..n_exts]) |props| {
                const name = std.mem.sliceTo(&props.extension_name, 0);
                if (ext_set.contains(name)) n_matched += 1;
            }
            if (n_matched < opts.vulkan_device_extensions.len) {
                continue;
            }
            std.debug.assert(n_matched == opts.vulkan_device_extensions.len);

            // Check queue families
            var n_queues: u32 = undefined;
            self.vki.getPhysicalDeviceQueueFamilyProperties(dev, &n_queues, null);
            const queues = try allocator.alloc(vk.QueueFamilyProperties, n_queues);
            defer allocator.free(queues);
            self.vki.getPhysicalDeviceQueueFamilyProperties(dev, &n_queues, queues.ptr);

            self.queue_family = for (queues[0..n_queues], 0..) |queue, i| {
                if (queue.queue_flags.compute_bit) {
                    break @intCast(i);
                }
            } else {
                continue;
            };

            break dev;
        } else {
            return error.NoSuitableDevice;
        };

        const props = self.vki.getPhysicalDeviceProperties(self.phys_device);
        self.device_properties = props;
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

        self.device = try self.vki.createDevice(self.phys_device, &vk.DeviceCreateInfo{
            .flags = .{},
            .queue_create_info_count = queue_infos.len,
            .p_queue_create_infos = &queue_infos,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = @intCast(opts.vulkan_device_extensions.len),
            .pp_enabled_extension_names = opts.vulkan_device_extensions.ptr,
            .p_enabled_features = &opts.vulkan_device_features,
        }, null);
        self.vkd = try DeviceDispatch.load(self.device, self.vki.dispatch.vkGetDeviceProcAddr);
        errdefer self.vkd.destroyDevice(self.device, null);

        self.queue = self.vkd.getDeviceQueue(self.device, self.queue_family, 0);
    }

    fn alloc(self: *Context, size: u64, mem_type_bits: u32, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        if (self.alloc_count >= self.device_properties.limits.max_memory_allocation_count) {
            return error.OutOfDeviceMemory;
        }
        const mem_type_idx = self.findMemoryType(size, mem_type_bits, flags) orelse {
            return error.UnsupportedAllocationFlags;
        };
        const mem = try self.vkd.allocateMemory(self.device, &.{
            .allocation_size = size,
            .memory_type_index = mem_type_idx,
        }, null);
        self.alloc_count += 1;
        return mem;
    }
    fn free(self: *Context, mem: vk.DeviceMemory) void {
        self.alloc_count -= 1;
        self.vkd.freeMemory(self.device, mem, null);
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
                return @intCast(mem_type_idx);
            }
        }
        return null;
    }
};

pub fn Shader(comptime parameter_decls: []const ShaderParameter) type {
    const param_info = ShaderParamInfo.init(parameter_decls);

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

        // Creates a shader from an array of native-endian u32. The default entrypoint name is `main`
        pub fn init(ctx: *Context, code: []const u32) !Self {
            return initNamed(ctx, code, "main");
        }

        // Creates a shader from an array of native-endian u32
        pub fn initNamed(ctx: *Context, code: []const u32, entrypoint_name: [:0]const u8) !Self {
            const module = try ctx.vkd.createShaderModule(ctx.device, &.{
                .flags = .{},
                .code_size = 4 * code.len,
                .p_code = code.ptr,
            }, null);
            defer ctx.vkd.destroyShaderModule(ctx.device, module, null);

            // TODO: cache descriptor sets?
            const desc_layout = try ctx.vkd.createDescriptorSetLayout(ctx.device, &vk.DescriptorSetLayoutCreateInfo{
                .flags = .{},
                .binding_count = @intCast(param_info.desc_layout_bindings.len),
                .p_bindings = param_info.desc_layout_bindings.ptr,
            }, null);
            errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.device, desc_layout, null);

            const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.device, &.{
                .flags = .{},
                .set_layout_count = 1,
                .p_set_layouts = &[_]vk.DescriptorSetLayout{desc_layout},
                .push_constant_range_count = param_info.push_constant_ranges.len,
                .p_push_constant_ranges = param_info.push_constant_ranges.ptr,
            }, null);
            errdefer ctx.vkd.destroyPipelineLayout(ctx.device, pipeline_layout, null);

            var pipeline: [1]vk.Pipeline = undefined;
            _ = try ctx.vkd.createComputePipelines(
                ctx.device,
                .null_handle, // TODO: pipeline caching?
                1,
                &[_]vk.ComputePipelineCreateInfo{.{
                    .flags = .{ .dispatch_base_bit = true },
                    .stage = .{
                        .flags = .{},
                        .stage = .{ .compute_bit = true },
                        .module = module,
                        .p_name = entrypoint_name.ptr,
                        .p_specialization_info = null,
                    },
                    .layout = pipeline_layout,
                    .base_pipeline_handle = .null_handle,
                    .base_pipeline_index = 0,
                }},
                null,
                &pipeline,
            );
            errdefer ctx.vkd.destroyPipeline(ctx.device, pipeline[0], null);

            const cmd_pool = try ctx.vkd.createCommandPool(ctx.device, &.{
                .flags = .{
                    .transient_bit = true,
                    .reset_command_buffer_bit = true,
                },
                .queue_family_index = ctx.queue_family,
            }, null);
            errdefer ctx.vkd.destroyCommandPool(ctx.device, cmd_pool, null);

            const desc_pool = try ctx.vkd.createDescriptorPool(ctx.device, &.{
                .flags = .{},
                .max_sets = 2,
                .pool_size_count = param_info.desc_pool_sizes.len,
                .p_pool_sizes = param_info.desc_pool_sizes.ptr,
            }, null);
            errdefer ctx.vkd.destroyDescriptorPool(ctx.device, desc_pool, null);

            const desc_template = try ctx.vkd.createDescriptorUpdateTemplate(ctx.device, &.{
                .flags = .{},
                .descriptor_update_entry_count = param_info.desc_template_entries.len,
                .p_descriptor_update_entries = param_info.desc_template_entries.ptr,
                .template_type = .descriptor_set,
                .descriptor_set_layout = desc_layout,
                // These fields are ignored in Vulkan 1.1, but we have them so we might as well provide them
                .pipeline_bind_point = .compute,
                .pipeline_layout = pipeline_layout,
                .set = undefined, // idk what this one is
            }, null);
            errdefer ctx.vkd.destroyDescriptorUpdateTemplate(ctx.device, desc_template, null);

            const fence = try ctx.vkd.createFence(ctx.device, &.{
                .flags = .{ .signaled_bit = true },
            }, null);
            errdefer ctx.vkd.destroyFence(ctx.device, fence, null);

            var cmd_bufs: [2]vk.CommandBuffer = undefined;
            try ctx.vkd.allocateCommandBuffers(ctx.device, &.{
                .command_pool = cmd_pool,
                .level = .primary,
                .command_buffer_count = 2,
            }, &cmd_bufs);

            var desc_sets: [2]vk.DescriptorSet = undefined;
            try ctx.vkd.allocateDescriptorSets(ctx.device, &.{
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
        // NOTE: the allocator is used for temporary allocations during setup, it is not needed after initBytes returns
        pub fn initBytes(allocator: std.mem.Allocator, ctx: *Context, code: []const u8) !Self {
            if (code.len & 3 != 0 or code.len == 0) {
                return error.InvalidShader;
            }

            // Detect endianness
            const magic = std.mem.readInt(u32, code[0..4], .little);
            const spirv_magic: u32 = 0x07230203;
            const endian: std.builtin.Endian = switch (magic) {
                spirv_magic => .little,
                @byteSwap(spirv_magic) => .big,
                else => return error.InvalidShader,
            };

            // Read SPIR-V
            const code32 = try allocator.alloc(u32, @divExact(code.len, 4));
            defer allocator.free(code32);
            for (code32, 0..) |*v, i| {
                v.* = std.mem.readInt(u32, code[i * 4 ..][0..4], endian);
            }

            // Init shader
            return init(ctx, code32);
        }

        pub fn deinit(self: Self) void {
            self.ctx.vkd.destroyFence(self.ctx.device, self.fence, null);
            self.ctx.vkd.destroyDescriptorUpdateTemplate(self.ctx.device, self.desc_template, null);
            self.ctx.vkd.destroyDescriptorPool(self.ctx.device, self.desc_pool, null);
            self.ctx.vkd.destroyCommandPool(self.ctx.device, self.cmd_pool, null);
            self.ctx.vkd.destroyPipeline(self.ctx.device, self.pipeline, null);
            self.ctx.vkd.destroyPipelineLayout(self.ctx.device, self.pipeline_layout, null);
            self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.device, self.desc_layout, null);
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
            param_data: Params,
        ) !void {
            // Update descriptor set
            const desc_set = self.desc_sets[self.idx];

            var descriptors: [param_info.desc_binding_names.len]vk.DescriptorBufferInfo = undefined;
            inline for (param_info.desc_binding_names, 0..) |name, i| {
                descriptors[i] = .{
                    .buffer = @field(param_data, name).buf,
                    .offset = 0,
                    .range = vk.WHOLE_SIZE,
                };
            }
            self.ctx.vkd.updateDescriptorSetWithTemplate(
                self.ctx.device,
                desc_set,
                self.desc_template,
                @ptrCast(&descriptors),
            );

            // Record command buffer
            const cmd_buf = self.cmd_bufs[self.idx];
            try self.ctx.vkd.beginCommandBuffer(cmd_buf, &.{
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

            // Send push constants
            inline for (param_info.push_constant_names, 0..) |name, i| {
                const push_data = @field(param_data, name);
                const range = param_info.push_constant_ranges[i];
                comptime std.debug.assert(range.size == @sizeOf(@TypeOf(push_data)));

                self.ctx.vkd.cmdPushConstants(
                    cmd_buf,
                    self.pipeline_layout,
                    range.stage_flags,
                    range.offset,
                    range.size,
                    @ptrCast(&push_data),
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

        pub const Params = param_info.params_type;
    };
}
pub const ComputeDispatch = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
    baseX: u32 = 0,
    baseY: u32 = 0,
    baseZ: u32 = 0,
};

pub fn uniformBuffer(comptime name: [:0]const u8, comptime binding: u32, comptime data_type: type) ShaderParameter {
    return .{
        .name = name,
        .idx = binding,
        .kind = .uniform,
        .data_type = data_type,
    };
}
pub fn storageBuffer(comptime name: [:0]const u8, comptime binding: u32, comptime data_type: type) ShaderParameter {
    return .{
        .name = name,
        .idx = binding,
        .kind = .storage,
        .data_type = data_type,
    };
}
pub fn pushConstant(comptime name: [:0]const u8, comptime offset: u32, comptime data_type: type) ShaderParameter {
    return .{
        .name = name,
        .idx = offset,
        .kind = .push_constant,
        .data_type = data_type,
    };
}

pub const ShaderParameter = struct {
    name: [:0]const u8,
    idx: u32, // Binding index for storage and uniform, offset for push_constant
    kind: ShaderParameterKind,
    data_type: type,
};
pub const ShaderParameterKind = enum {
    uniform,
    storage,
    push_constant,

    fn toVk(t: ShaderParameterKind) vk.DescriptorType {
        return switch (t) {
            .uniform => .uniform_buffer,
            .storage => .storage_buffer,
            .push_constant => unreachable,
        };
    }
};

const ShaderParamInfo = struct {
    push_constant_names: []const []const u8,
    push_constant_ranges: []const vk.PushConstantRange,

    desc_binding_names: []const []const u8,
    desc_layout_bindings: []const vk.DescriptorSetLayoutBinding,
    desc_template_entries: []const vk.DescriptorUpdateTemplateEntry,
    desc_pool_sizes: []const vk.DescriptorPoolSize,

    params_type: type,

    fn init(comptime params: []const ShaderParameter) ShaderParamInfo {
        var kind_counts = std.EnumArray(ShaderParameterKind, u32).initDefault(0, .{});
        var fields: [params.len]std.builtin.Type.StructField = undefined;

        var push_constant_names: [params.len][]const u8 = undefined;
        var push_constant_ranges: [params.len]vk.PushConstantRange = undefined;
        var push_constant_i = 0;

        var desc_binding_names: [params.len][]const u8 = undefined;
        var desc_layout_bindings: [params.len]vk.DescriptorSetLayoutBinding = undefined;
        var desc_template_entries: [params.len]vk.DescriptorUpdateTemplateEntry = undefined;
        var desc_i = 0;

        for (params, 0..) |param, i| {
            fields[i] = .{
                .name = param.name,
                .type = param.data_type,
                .default_value = null,
                .is_comptime = false,
                .alignment = @alignOf(param.data_type),
            };

            if (param.kind == .push_constant) {
                switch (@typeInfo(param.data_type)) {
                    .Struct => |info| if (info.layout == .auto) {
                        @compileError("Push constant data should have defined layout. Use an extern struct (or a packed struct if you know what you're doing)");
                    },
                    .Union => |info| if (info.layout == .auto) {
                        @compileError("Push constant data should have defined layout. Unions are a bad idea anyway, but if you must, use an extern or packed union");
                    },
                    // TODO: there's a lot more checking we could do here but I'm lazy
                    else => {},
                }

                push_constant_names[push_constant_i] = param.name;

                push_constant_ranges[push_constant_i] = .{
                    .stage_flags = .{ .compute_bit = true },
                    .offset = param.idx,
                    .size = @sizeOf(param.data_type),
                };

                push_constant_i += 1;
            } else {
                if (!isBufferWrapper(param.data_type)) {
                    @compileError("Shader bindings must use buffer types");
                }

                desc_binding_names[desc_i] = param.name;

                desc_layout_bindings[desc_i] = .{
                    .binding = param.idx,
                    .descriptor_type = param.kind.toVk(),
                    .descriptor_count = 1,
                    .stage_flags = .{ .compute_bit = true },
                    .p_immutable_samplers = null,
                };

                // TODO: merge consecutive bindings
                desc_template_entries[desc_i] = .{
                    .dst_binding = param.idx,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = param.kind.toVk(),
                    .offset = desc_i * @sizeOf(vk.DescriptorBufferInfo),
                    .stride = 0, // TODO
                };

                kind_counts.getPtr(param.kind).* += 1;

                desc_i += 1;
            }
        }

        var pool_sizes: [kind_counts.values.len]vk.DescriptorPoolSize = undefined;
        var pool_i = 0;
        var it = kind_counts.iterator();
        while (it.next()) |entry| {
            if (entry.value.* > 0) {
                pool_sizes[pool_i] = .{
                    .type = entry.key.toVk(),
                    // Multiply by 2 because we need two descriptor sets
                    .descriptor_count = 2 * entry.value.*,
                };
                pool_i += 1;
            }
        }

        // We copy the arrays in order to remove unused elements from the binary
        return .{
            .push_constant_names = constSlice(
                []const u8,
                push_constant_names[0..push_constant_i],
            ),
            .push_constant_ranges = constSlice(
                vk.PushConstantRange,
                push_constant_ranges[0..push_constant_i],
            ),

            .desc_binding_names = constSlice(
                []const u8,
                desc_binding_names[0..desc_i],
            ),
            .desc_layout_bindings = constSlice(
                vk.DescriptorSetLayoutBinding,
                desc_layout_bindings[0..desc_i],
            ),
            .desc_template_entries = constSlice(
                vk.DescriptorUpdateTemplateEntry,
                desc_template_entries[0..desc_i],
            ),
            .desc_pool_sizes = constSlice(
                vk.DescriptorPoolSize,
                pool_sizes[0..pool_i],
            ),

            .params_type = @Type(.{ .Struct = .{
                .layout = .auto,
                .fields = &fields,
                .decls = &.{},
                .is_tuple = false,
            } }),
        };
    }
};

/// Given a comptime-known slice, stores the data in a `const` and returns a
/// pointer to it, hence giving a slice which is available at runtime.
fn constSlice(comptime T: type, comptime x: []const T) []const T {
    const final: [x.len]T = x[0..].*;
    return &final;
}

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
            const buf = try ctx.vkd.createBuffer(ctx.device, &vk.BufferCreateInfo{
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
            }, null);
            errdefer ctx.vkd.destroyBuffer(ctx.device, buf, null);

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
            self.ctx.vkd.destroyBuffer(self.ctx.device, self.buf, null);
        }

        const min_map_align = 64; // Spec requires min_memory_map_alignment limit to be at least 64
        pub fn map(self: Self) ![]align(min_map_align) T {
            return self.mapRange(0, self.len);
        }
        pub fn mapRange(self: Self, off: u64, len: u64) ![]align(min_map_align) T {
            const ptr = try self.ctx.vkd.mapMemory(
                self.ctx.device,
                self.mem,
                self.off + off,
                len * @sizeOf(T),
                .{},
            );
            const ptr_typed: [*]align(min_map_align) T = @alignCast(@ptrCast(ptr));
            return ptr_typed[0..self.len];
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
        std.meta.fieldInfo(T, .buf).type == vk.Buffer and
        @hasDecl(T, "is_zcompute_buffer_wrapper");
}

const need_api: vk.ApiInfo = .{
    .base_commands = .{
        .createInstance = true,
        .enumerateInstanceLayerProperties = true,
        .getInstanceProcAddr = true,
    },
    .instance_commands = .{
        .createDevice = true,
        .destroyInstance = true,
        .enumerateDeviceExtensionProperties = true,
        .enumeratePhysicalDevices = true,
        .getDeviceProcAddr = true,
        .getPhysicalDeviceFeatures = true,
        .getPhysicalDeviceMemoryProperties = true,
        .getPhysicalDeviceProperties = true,
        .getPhysicalDeviceQueueFamilyProperties = true,
    },
    .device_commands = .{
        .allocateCommandBuffers = true,
        .allocateDescriptorSets = true,
        .allocateMemory = true,
        .beginCommandBuffer = true,
        .bindBufferMemory = true,
        .cmdBindDescriptorSets = true,
        .cmdBindPipeline = true,
        .cmdDispatchBase = true,
        .cmdPushConstants = true,
        .createBuffer = true,
        .createCommandPool = true,
        .createComputePipelines = true,
        .createDescriptorPool = true,
        .createDescriptorSetLayout = true,
        .createDescriptorUpdateTemplate = true,
        .createFence = true,
        .createPipelineLayout = true,
        .createShaderModule = true,
        .destroyBuffer = true,
        .destroyCommandPool = true,
        .destroyDescriptorPool = true,
        .destroyDescriptorSetLayout = true,
        .destroyDescriptorUpdateTemplate = true,
        .destroyDevice = true,
        .destroyFence = true,
        .destroyPipeline = true,
        .destroyPipelineLayout = true,
        .destroyShaderModule = true,
        .endCommandBuffer = true,
        .freeMemory = true,
        .getBufferMemoryRequirements = true,
        .getDeviceQueue = true,
        .mapMemory = true,
        .queueSubmit = true,
        .resetFences = true,
        .unmapMemory = true,
        .updateDescriptorSetWithTemplate = true,
        .waitForFences = true,
    },
};

const BaseDispatch = vk.BaseWrapper(&.{need_api});
const InstanceDispatch = vk.InstanceWrapper(&.{need_api});
const DeviceDispatch = vk.DeviceWrapper(&.{need_api});

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
            .macos => @compileError("Unsupported platform: " ++ @tagName(builtin.os.tag)),
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
