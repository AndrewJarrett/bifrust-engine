use crate::app::App;
use crate::device::BfDevice;
use crate::window::BfWindow;
use crate::swapchain::Swapchain;

use std::ptr::slice_from_raw_parts;
use std::mem::size_of;

use anyhow::{anyhow, Result};
use cgmath::{vec3, Deg};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;

//type Vec2 = cgmath::Vector2<f32>;
//type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;
//type Pt3  = cgmath::Point3<f32>;

pub struct Renderer {
    pub swapchain: Swapchain,
    pub is_frame_started: bool,
    pub image_index: usize,
    pub current_frame_index: usize,
    pub command_buffer: Option<vk::CommandBuffer>,
}

impl Renderer {

    pub fn new(bf_window: &BfWindow, bf_device: &mut BfDevice) -> Result<Self> {
        let swapchain: Swapchain;
        unsafe {
            swapchain = Swapchain::new(&bf_window, &bf_device)?;
        }

        bf_device.create_command_buffers(&swapchain)?;

        let is_frame_started = false;

        Ok(Self {
            swapchain,
            is_frame_started,
            image_index: 0,
            current_frame_index: 0,
            command_buffer: None,
        })
    }

    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, bf_device: &BfDevice, bf_window: &BfWindow, app: &App) -> Result<()> {
        bf_device.device.device_wait_idle()?;
        self.destroy_swapchain(&bf_device, &app);

        self.swapchain = Swapchain::new(&bf_window, &bf_device)?;

        Ok(())
    }

    pub unsafe fn begin_frame(&mut self, bf_device: &BfDevice, bf_window: &BfWindow, app: &App) -> Result<()> {

        let result = self.swapchain.acquire_next_image(&bf_device)?;

        let image_index = match result { 
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(
                &bf_device,
                &bf_window,
                &app
            ),
            Err(e) => return Err(anyhow!(e)),
        };

        self.is_frame_started = true;
        self.image_index = image_index;

        self.update_command_buffer(&bf_device)?;

        Ok(())
    }

    unsafe fn update_command_buffer(
        &mut self,
        bf_device: &BfDevice,
    ) -> Result<()> {
        // Reset command pool
        let command_pool = bf_device.command_pools[self.image_index];
        bf_device.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        self.command_buffer = Some(bf_device.command_buffers[self.image_index]);

        let info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        bf_device.device.begin_command_buffer(self.command_buffer.unwrap(), &info)?;

        Ok(())
    }

    pub unsafe fn begin_swapchain_render_pass(
        &mut self,
        bf_device: &mut BfDevice,
        app: &App,
    ) -> Result<()> {
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.swapchain.extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.swapchain.render_pass)
            .framebuffer(self.swapchain.framebuffers[self.image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        bf_device.device.cmd_begin_render_pass(
            self.command_buffer.unwrap(),
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
        );

        let secondary_command_buffers = (0..app.models)
            .map(|i| self.update_secondary_command_buffer(bf_device, &app, i))
            .collect::<Result<Vec<_>, _>>()?;
        bf_device.device.cmd_execute_commands(self.command_buffer.unwrap(), &secondary_command_buffers[..]);

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        bf_device: &mut BfDevice,
        app: &App,
        model_index: usize
    ) -> Result<vk::CommandBuffer> {
        bf_device.secondary_command_buffers.resize_with(self.image_index + 1, Vec::new);

        let command_buffers = &mut bf_device.secondary_command_buffers[self.image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(bf_device.command_pools[self.image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = bf_device.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // Update model
        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let model = Mat4::from_translation(vec3(0.0, y, z)) * Mat4::from_axis_angle(
            vec3(0.0, 0.0, 1.0),
            Deg(90.0) * app.delta_time
        );

        let model_bytes = &*slice_from_raw_parts(&model as *const Mat4 as *const u8, size_of::<Mat4>());

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.swapchain.render_pass)
            .subpass(0)
            .framebuffer(self.swapchain.framebuffers[self.image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        bf_device.device.begin_command_buffer(command_buffer, &info)?;

        bf_device.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app.data.pipeline
        );
        bf_device.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &[app.data.vertex_buffer],
            &[0]
        );
        bf_device.device.cmd_bind_index_buffer(
            command_buffer,
            app.data.index_buffer,
            0,
            vk::IndexType::UINT32
        );
        bf_device.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            app.data.pipeline_layout,
            0,
            &[app.data.descriptor_sets[self.image_index]],
            &[],
        );
        bf_device.device.cmd_push_constants(
            command_buffer,
            app.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        bf_device.device.cmd_push_constants(
            command_buffer,
            app.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        bf_device.device.cmd_draw_indexed(
            command_buffer,
            app.data.indices.len() as u32,
            1,
            0,
            0,
            0
        );

        Ok(command_buffer)
    }

    pub unsafe fn end_swapchain_render_pass(
        &mut self,
        bf_device: &BfDevice,
    ) -> Result<()> {
        bf_device.device.cmd_end_render_pass(self.command_buffer.unwrap());

        Ok(())
    }

    pub unsafe fn end_frame(
        &mut self,
        bf_device: &BfDevice,
        bf_window: &BfWindow,
        app: &mut App,
    ) -> Result<()> {
        let result = self.swapchain.submit_command_buffers(&bf_device, self.image_index)?;

        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if app.resized || changed {
            app.resized = false;
            self.recreate_swapchain(&bf_device, &bf_window, &app)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.is_frame_started = false;
        self.current_frame_index = (self.current_frame_index + 1) % crate::swapchain::MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    pub fn get_aspect_ratio(&self) -> Result<f32> {
        Ok(self.swapchain.get_extent_aspect_ratio()?)
    }

    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&self, bf_device: &BfDevice, app: &App) {
        bf_device.device.destroy_descriptor_pool(app.data.descriptor_pool, None);
        app.data.uniform_buffers_memory.iter().for_each(|m| bf_device.device.free_memory(*m, None));
        app.data.uniform_buffers.iter().for_each(|b| bf_device.device.destroy_buffer(*b, None));
        bf_device.device.destroy_image_view(self.swapchain.depth_image_view, None);
        bf_device.device.free_memory(self.swapchain.depth_image_memory, None);
        bf_device.device.destroy_image(self.swapchain.depth_image, None);
        bf_device.device.destroy_image_view(app.data.color_image_view, None);
        bf_device.device.free_memory(app.data.color_image_memory, None);
        bf_device.device.destroy_image(app.data.color_image, None);
        self.swapchain.framebuffers.iter().for_each(|f| bf_device.device.destroy_framebuffer(*f, None));
        bf_device.device.destroy_pipeline(app.data.pipeline, None);
        bf_device.device.destroy_pipeline_layout(app.data.pipeline_layout, None);
        bf_device.device.destroy_render_pass(self.swapchain.render_pass, None);
        self.swapchain.image_views.iter().for_each(|v| bf_device.device.destroy_image_view(*v, None));
        bf_device.device.destroy_swapchain_khr(self.swapchain.swapchain, None);
    }

}
