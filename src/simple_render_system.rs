use crate::device::BfDevice;
use crate::pipeline::{Pipeline, PipelineConfigInfo};
use crate::swapchain::Swapchain;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct SimpleRenderSystem {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

impl SimpleRenderSystem {

    pub unsafe fn new(
        bf_device: &BfDevice,
        swapchain: &Swapchain,
    ) -> Result<Self> {

        let pipeline_layout = Self::create_pipeline_layout(&bf_device)?;
        let pipeline = Self::create_pipeline(&bf_device, &swapchain, pipeline_layout)?;

        Ok(Self {
            pipeline,
            pipeline_layout,
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

    unsafe fn create_pipeline(
        bf_device: &BfDevice,
        swapchain: &Swapchain,
        pipeline_layout: vk::PipelineLayout
    ) -> Result<vk::Pipeline> {
        let pipeline_config_info = PipelineConfigInfo::new(
            &bf_device,
            &swapchain,
            pipeline_layout
        )?;

        let pipeline = Pipeline::new(&bf_device, pipeline_config_info)?;
        Ok(pipeline.pipeline)
    }
}
