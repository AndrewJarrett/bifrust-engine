#![allow(
    dead_code,
    unused_variables
)]

use crate::device::BfDevice;
use crate::app::Vertex;
use crate::swapchain::Swapchain;

use anyhow::Result;
//use log::*;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::bytecode::Bytecode;
//use vulkanalia::window as vk_window;

#[derive(Debug)]
pub struct Pipeline {
    vert: Vec<u8>,
    frag: Vec<u8>,
    config_info: PipelineConfigInfo,
    pub pipeline: vk::Pipeline,
    vert_shader_module: vk::ShaderModule,
    frag_shader_module: vk::ShaderModule,
}

impl Pipeline {

    pub unsafe fn new(
        bf_device: &BfDevice,
        config_info: PipelineConfigInfo
    ) -> Result<Self> {
        let vert = include_bytes!("../shaders/vert.spv").to_vec();
        let frag = include_bytes!("../shaders/frag.spv").to_vec();

        let vert_shader_module = Self::create_shader_module(&bf_device, &vert[..])?;
        let frag_shader_module = Self::create_shader_module(&bf_device, &frag[..])?;

        let pipeline = Self::create_graphics_pipeline(&bf_device, vert_shader_module, frag_shader_module, &config_info)?;

        Ok(Self {
            vert,
            frag,
            config_info,
            pipeline,
            vert_shader_module,
            frag_shader_module,
        })
    }

    unsafe fn create_graphics_pipeline(
        bf_device: &BfDevice,
        vert_shader_module: vk::ShaderModule,
        frag_shader_module: vk::ShaderModule, 
        config_info: &PipelineConfigInfo
    ) -> Result<vk::Pipeline> {

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(b"main\0");

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(b"main\0");

        let binding_descriptions = &[Vertex::binding_description()];
        let attribute_descriptions = Vertex::attribute_descriptions();
        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);
        
        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&config_info.input_assembly_state)
            .viewport_state(&config_info.viewport_state)
            .rasterization_state(&config_info.rasterization_state)
            .multisample_state(&config_info.multisample_state)
            .depth_stencil_state(&config_info.depth_stencil_state)
            .color_blend_state(&config_info.color_blend_state)
            .layout(config_info.pipeline_layout)
            .render_pass(config_info.render_pass)
            .subpass(config_info.subpass);

        let pipeline = bf_device.device.create_graphics_pipelines(
            vk::PipelineCache::null(), &[info], None)?.0[0];

        bf_device.device.destroy_shader_module(vert_shader_module, None);
        bf_device.device.destroy_shader_module(frag_shader_module, None);

        Ok(pipeline)
    }

    unsafe fn create_shader_module(bf_device: &BfDevice, bytecode: &[u8]) -> Result<vk::ShaderModule> {
        let bytecode = Bytecode::new(bytecode).unwrap();

        let info = vk::ShaderModuleCreateInfo::builder()
            .code_size(bytecode.code_size())
            .code(bytecode.code());

        Ok(bf_device.device.create_shader_module(&info, None)?)
    }

    pub unsafe fn destroy(&self, bf_device: &BfDevice) -> Result<()> {
        bf_device.device.destroy_pipeline(self.pipeline, None);

        Ok(())
    }
}

#[derive(Debug)]
pub struct PipelineConfigInfo {
    viewport: vk::Viewport,
    scissor: vk::Rect2D,
    viewport_state: vk::PipelineViewportStateCreateInfo,
    input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo,
    rasterization_state: vk::PipelineRasterizationStateCreateInfo,
    multisample_state: vk::PipelineMultisampleStateCreateInfo,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    color_blend_state: vk::PipelineColorBlendStateCreateInfo,
    depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    subpass: u32,
}

impl PipelineConfigInfo {

    pub unsafe fn new(
        bf_device: &BfDevice,
        swapchain: &Swapchain,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<PipelineConfigInfo> {
        let width = swapchain.extent.width;
        let height = swapchain.extent.height;

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D { width, height });

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(true)
            .min_sample_shading(0.2)
            .rasterization_samples(bf_device.msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[color_blend_attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_states = &[
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::LINE_WIDTH,
        ];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(dynamic_states);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false);

        Ok(Self {
            viewport: viewport.build(),
            scissor: scissor.build(),
            viewport_state: viewport_state.build(),
            input_assembly_state: input_assembly_state.build(),
            rasterization_state: rasterization_state.build(),
            multisample_state: multisample_state.build(),
            color_blend_attachment: color_blend_attachment.build(),
            color_blend_state: color_blend_state.build(),
            depth_stencil_state: depth_stencil_state.build(),
            pipeline_layout,
            render_pass: swapchain.render_pass,
            subpass: 0,
        })
    }

    unsafe fn create_pipeline_layout(bf_device: &BfDevice) -> Result<vk::PipelineLayout> {
        let vert_push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64);

        let frag_push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(64)
            .size(4);

        let descriptor_set_layout = Self::create_descriptor_set_layout(&bf_device)?;

        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);

        Ok(bf_device.device.create_pipeline_layout(&layout_info, None)?)
    }


    unsafe fn create_descriptor_set_layout(bf_device: &BfDevice) -> Result<vk::DescriptorSetLayout> {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding, sampler_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

        Ok(bf_device.device.create_descriptor_set_layout(&info, None)?)
    }
}
