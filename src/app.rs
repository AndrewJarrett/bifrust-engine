#![allow(
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use crate::window::{BfWindow, BfWindowData};
use crate::device::BfDevice;
use crate::pipeline::{Pipeline, PipelineConfigInfo};
use crate::swapchain::{Swapchain};

use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::BufReader;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;
use std::ptr::slice_from_raw_parts;
use std::f64::consts::FRAC_PI_2;

use anyhow::{anyhow, Result};
use cgmath::{vec2, vec3, point3, Deg};
use log::*;
use thiserror::Error;
use vulkanalia::prelude::v1_0::*;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
};

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSwapchainExtension;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const VELOCITY: f32 = 0.1;
const MOUSE_SPEED: f32 = 0.005;

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;
type Pt3  = cgmath::Point3<f32>;

/// Our Vulkan app.
#[derive(Clone, Debug)]
pub struct App {
    bf_device: BfDevice,
    //pub instance: Instance,
    pub data: AppData,
    pub bf_window_data: BfWindowData,
    //pub device: Device,
    frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub delta_time: f32,
    pub models: usize,
    pub position: Pt3,
    pub direction: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub vertical_angle: f32,
    pub horizontal_angle: f32,
    pub delta_mouse: (f64, f64),
}

impl App {

    #[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
    pub unsafe fn run() {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                std::panic::set_hook(Box::new(console_error_panic_hook::hook));
                console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
            } else {
                pretty_env_logger::init();
            }
        }

        // Window
        let event_loop = EventLoop::new().unwrap();
        let bf_window = BfWindow::new(1024, 768, "Bifrust Engine Tester".to_string(), &event_loop).unwrap();

        // App
        let mut app = unsafe { App::create(&bf_window).unwrap() };

        let extension_count = unsafe { app.get_extension_count() };
        info!("Found {} physical device extension!", extension_count);

        event_loop.set_control_flow(ControlFlow::Poll);

        let _ = event_loop.run(move |event, elwt| {
            match event {
                // Request a redraw to render continuously
                Event::AboutToWait if !app.bf_window_data.minimized => {
                    if app.bf_window_data.destroying {
                        elwt.exit();
                    } else {
                        bf_window.window.request_redraw();
                    }
                }
                // Render a frame if our Vulkan app is not being destroyed.
                Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                    if app.bf_window_data.destroying { println!("Redraw") };
                    app.delta_time = app.start.elapsed().as_secs_f32();

                    unsafe { app.render(&bf_window) }.unwrap();
                }
                // Handle user input - keyboard
                Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
                    if app.bf_window_data.destroying { println!("Keyboard") };
                    if event.state == ElementState::Pressed {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowLeft) if app.models > 1 => app.models -= 1,
                            PhysicalKey::Code(KeyCode::ArrowRight) if app.models < 4 => app.models += 1,
                            PhysicalKey::Code(KeyCode::KeyW) => app.update_position(KeyCode::KeyW),
                            PhysicalKey::Code(KeyCode::KeyA) => app.update_position(KeyCode::KeyA), 
                            PhysicalKey::Code(KeyCode::KeyS) => app.update_position(KeyCode::KeyS), 
                            PhysicalKey::Code(KeyCode::KeyD) => app.update_position(KeyCode::KeyD), 
                            // Escape from the app
                            PhysicalKey::Code(KeyCode::Escape) => {
                                app.bf_window_data.destroying = true;
                            }
                            _ => { }
                        }
                    }
                }
                // Mouse movement
                Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                    if app.bf_window_data.destroying { println!("Mouse") };
                    app.delta_mouse = delta;
                    //dbg!(app.delta_mouse);
                }
                // Handle window is resized
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    if size.width == 0 || size.height == 0 {
                        app.bf_window_data.minimized = true;
                    } else {
                        app.bf_window_data.minimized = false;
                        app.resized = true;
                    }
                }
                // Destroy our Vulkan app
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    app.bf_window_data.destroying = true;
                    elwt.exit();
                    unsafe { app.bf_device.device.device_wait_idle().unwrap(); }
                    unsafe { app.destroy(&bf_window); }
                }
                _ => {}
            }
        });
    }

    /// Creates our Vulkan app.
    pub unsafe fn create(bf_window: &BfWindow) -> Result<Self> {
        let bf_window_data = BfWindowData::default();
        let mut data = AppData::default();

        let (indices, _vertices) = Self::load_model()?;
        data.indices = indices;

        println!("Made it here 1!");
        let mut bf_device = BfDevice::new(&bf_window)?;

        println!("Made it here 2!");
        let swapchain = Swapchain::new(&bf_window, &bf_device)?;

        //create_swapchain(&bf_window, &bf_device, &mut data);
        //create_swapchain_image_views(&bf_device.device, &mut data)?;
        //create_render_pass(&bf_device.instance, &bf_device.device, &mut data)?;
        //create_descriptor_set_layout(&bf_device.device, &mut data)?;

        println!("Made it here 3!");
        let pipeline_config_info = PipelineConfigInfo::new(&bf_device, &swapchain, bf_window.width, bf_window.height)?;

        let pipeline = Pipeline::new(&bf_device, pipeline_config_info)?;
        data.pipeline = pipeline.pipeline;

        //create_pipeline(&device, &mut data)?;
        //create_command_pools(&instance, &device, &mut data, &bf_device.surface)?;
        create_color_objects(&bf_device.instance, &bf_device.device, &mut data)?;
        //create_depth_objects(&bf_device.instance, &bf_device.device, &mut data)?;
        //create_framebuffers(&bf_device.device, &mut data)?;
        create_texture_image(&bf_device.instance, &bf_device.device, &mut data)?;
        create_texture_image_view(&bf_device.device, &mut data)?;
        create_texture_sampler(&bf_device.device, &mut data)?;
        //create_vertex_buffer(&bf_device.instance, &bf_device.device, &mut data)?;
        create_index_buffer(&bf_device.instance, &bf_device.device, &mut data)?;
        create_uniform_buffers(&bf_device.instance, &bf_device.device, &mut data)?;
        create_descriptor_pool(&bf_device.device, &mut data)?;
        create_descriptor_sets(&bf_device.device, &mut data)?;

        bf_device.create_command_buffers(&swapchain)?;

        //create_command_buffers(&bf_device.device, &mut data)?;
        //create_sync_objects(&bf_device.device, &mut data)?;
        Ok(Self { 
            bf_device,
            data,
            bf_window_data,
            frame: 0,
            resized: false,
            start: Instant::now(),
            delta_time: Instant::now().elapsed().as_secs_f32(),
            models: 1,
            position: point3::<f32>(6.0, 0.0, 2.0),
            direction: vec3::<f32>(0.0, 0.0, 0.0),
            right: vec3::<f32>(0.0, 0.0, 0.0),
            up: vec3::<f32>(0.0, 0.0, 0.0),
            vertical_angle: 0.0,
            horizontal_angle: 0.0,
            delta_mouse: (0.0, 0.0),
        })
    }

    /// Gets the number of Physical Device extensions
    pub unsafe fn get_extension_count(&self) -> u32 {
        let results = self.bf_device.instance
            .enumerate_device_extension_properties(self.data.physical_device, None)
            .unwrap();

        results.len() as u32
    }

    fn load_model() -> Result<(Vec<u32>, Vec<Vertex>)> {
        let mut reader = BufReader::new(File::open("resources/viking_room.obj")?);
        //let mut reader = BufReader::new(File::open("resources/orange-3d.obj")?);
        
        let mut indices: Vec<u32> = Vec::new();
        let mut vertices: Vec<Vertex> = Vec::new();

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions { triangulate: true, ..Default::default() },
            |_| Ok(Default::default()),
        )?;

        let mut unique_vertices = HashMap::new();

        for model in &models {
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = Vertex {
                    pos: vec3(
                        model.mesh.positions[pos_offset],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2],
                    ),
                    color: vec3(1.0, 1.0, 1.0),
                    tex_coord: vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                    ),
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();
                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
        }

        Ok((indices, vertices))
    }

    /// Renders a frame for our Vulkan app.
    pub unsafe fn render(&mut self, bf_window: &BfWindow) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.bf_device.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.bf_device.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result { 
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(&bf_window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.bf_device.device.wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = image_in_flight;

        self.update_command_buffer(image_index)?;
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.bf_device.device.reset_fences(&[in_flight_fence])?;

        self.bf_device.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.bf_device.device.queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(&bf_window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Reset command pool
        let command_pool = self.data.command_pools[image_index];
        self.bf_device.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.bf_device.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

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
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.bf_device.device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);

        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;
        self.bf_device.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        self.bf_device.device.cmd_end_render_pass(command_buffer);

        self.bf_device.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(&mut self, image_index: usize, model_index: usize) -> Result<vk::CommandBuffer> {
        self.data.secondary_command_buffers.resize_with(image_index + 1, Vec::new);

        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.bf_device.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // Update model
        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let model = Mat4::from_translation(vec3(0.0, y, z)) * Mat4::from_axis_angle(
            vec3(0.0, 0.0, 1.0),
            Deg(90.0) * self.delta_time
        );

        let model_bytes = &*slice_from_raw_parts(&model as *const Mat4 as *const u8, size_of::<Mat4>());

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.bf_device.device.begin_command_buffer(command_buffer, &info)?;

        self.bf_device.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);
        self.bf_device.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.vertex_buffer], &[0]);
        self.bf_device.device.cmd_bind_index_buffer(command_buffer, self.data.index_buffer, 0, vk::IndexType::UINT32);
        self.bf_device.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[],
        );
        self.bf_device.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );
        self.bf_device.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );
        self.bf_device.device.cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);

        self.bf_device.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let up = self.right.cross(self.direction);

        let _view_change = Mat4::look_at_rh(
            self.position,
            self.position + self.direction,
            up,
        );

        let view = Mat4::look_at_rh(
            point3(6.0, 0.0, 2.0),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        #[rustfmt::skip]
        let correction = Mat4::new(
            1.0, 0.0,       0.0, 0.0,
            // We're also flipping the Y-axis with this line's -1.0
            0.0, -1.0,      0.0, 0.0,
            0.0, 0.0, 1.0 / 2.0, 0.0,
            0.0, 0.0, 1.0 / 2.0, 1.0,
        );

        /*
        let proj_original = cgmath::perspective(
            Deg(45.0),
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,
            100.0,
        );
        */

        let proj = correction 
            * cgmath::perspective(
                Deg(45.0),
                self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
                0.1,
                200.0,
            );
        //dbg!(correction, proj_original, proj);

        let ubo = UniformBufferObject { view, proj };

        let memory = self.bf_device.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.bf_device.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    pub fn update_position(&mut self, key: KeyCode) -> () {
        let _direction = self.get_direction();
        let _right = self.get_right();

        match key {
            KeyCode::KeyW => self.position -= self.direction * VELOCITY * self.delta_time,
            KeyCode::KeyA => self.position += self.right * VELOCITY * self.delta_time,
            KeyCode::KeyS => self.position += self.direction * VELOCITY * self.delta_time,
            KeyCode::KeyD => self.position -= self.right * VELOCITY * self.delta_time,
            _ => {}
        }
    }

    fn get_direction(&mut self) -> Result<Vec3> {
        self.horizontal_angle    += MOUSE_SPEED * self.delta_time * (self.data.swapchain_extent.width as f32 / 2.0 - self.delta_mouse.0 as f32);
        self.vertical_angle      += MOUSE_SPEED * self.delta_time * (self.data.swapchain_extent.height as f32 / 2.0 - self.delta_mouse.1 as f32);

        let direction = vec3(
            self.vertical_angle.cos() * self.horizontal_angle.sin(),
            self.vertical_angle.sin(),
            self.vertical_angle.cos() * self.horizontal_angle.cos()
        );

        Ok(direction)
    }

    fn get_right(&mut self) -> Result<Vec3> {
        let right = vec3(
            (self.horizontal_angle - FRAC_PI_2 as f32).sin(),
            0.0,
            (self.horizontal_angle - FRAC_PI_2 as f32).cos()
        );

        Ok(right)
    }

    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, bf_window: &BfWindow) -> Result<()> {
        self.bf_device.device.device_wait_idle()?;
        self.destroy_swapchain();

        let mut swapchain = Swapchain::new(&bf_window, &self.bf_device)?;

        //create_swapchain(&bf_window, &self.bf_device, &mut self.data)?;
        //create_swapchain_image_views(&self.bf_device.device, &mut self.data)?;
        //create_render_pass(&self.bf_device.instance, &self.bf_device.device, &mut self.data)?;
        //create_pipeline(&self.bf_device.device, &mut self.data)?;
        let pipeline_config_info = PipelineConfigInfo::new(&self.bf_device, &swapchain, bf_window.width, bf_window.height)?;

        let pipeline = Pipeline::new(&self.bf_device, pipeline_config_info)?;
        self.data.pipeline = pipeline.pipeline;
        create_color_objects(&self.bf_device.instance, &self.bf_device.device, &mut self.data)?;
        //create_depth_objects(&self.bf_device.instance, &self.bf_device.device, &mut self.data)?;
        //create_framebuffers(&self.bf_device.device, &mut self.data)?;
        create_uniform_buffers(&self.bf_device.instance, &self.bf_device.device, &mut self.data)?;
        create_descriptor_pool(&self.bf_device.device, &mut self.data)?;
        create_descriptor_sets(&self.bf_device.device, &mut self.data)?;
        //create_command_buffers(&self.bf_device.device, &mut self.data)?;
        swapchain.sync.images_in_flight.resize(swapchain.images.len(), vk::Fence::null());

        Ok(())
    }

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    pub unsafe fn destroy(&mut self, bf_window: &BfWindow) {
        self.bf_device.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.data.in_flight_fences.iter().for_each(|f| self.bf_device.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores.iter().for_each(|s| self.bf_device.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores.iter().for_each(|s| self.bf_device.device.destroy_semaphore(*s, None));
        self.data.command_pools.iter().for_each(|p| self.bf_device.device.destroy_command_pool(*p, None));
        self.bf_device.device.free_memory(self.data.index_buffer_memory, None);
        self.bf_device.device.destroy_buffer(self.data.index_buffer, None);
        self.bf_device.device.free_memory(self.data.vertex_buffer_memory, None);
        self.bf_device.device.destroy_buffer(self.data.vertex_buffer, None);
        self.bf_device.device.destroy_sampler(self.data.texture_sampler, None);
        self.bf_device.device.destroy_image_view(self.data.texture_image_view, None);
        self.bf_device.device.free_memory(self.data.texture_image_memory, None);
        self.bf_device.device.destroy_image(self.data.texture_image, None);
        self.bf_device.device.destroy_command_pool(self.data.command_pool, None);
        self.bf_device.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.bf_device.device.destroy_device(None);

        // Destroy surface
        bf_window.destroy(&self.bf_device.instance, &self.bf_window_data).unwrap();

        if VALIDATION_ENABLED && self.bf_device.messenger.is_some() {
            self.bf_device.instance.destroy_debug_utils_messenger_ext(self.bf_device.messenger.unwrap(), None);
        }

        self.bf_device.instance.destroy_instance(None);
    }

    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.bf_device.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data.uniform_buffers_memory.iter().for_each(|m| self.bf_device.device.free_memory(*m, None));
        self.data.uniform_buffers.iter().for_each(|b| self.bf_device.device.destroy_buffer(*b, None));
        self.bf_device.device.destroy_image_view(self.data.depth_image_view, None);
        self.bf_device.device.free_memory(self.data.depth_image_memory, None);
        self.bf_device.device.destroy_image(self.data.depth_image, None);
        self.bf_device.device.destroy_image_view(self.data.color_image_view, None);
        self.bf_device.device.free_memory(self.data.color_image_memory, None);
        self.bf_device.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.bf_device.device.destroy_framebuffer(*f, None));
        self.bf_device.device.destroy_pipeline(self.data.pipeline, None);
        self.bf_device.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.bf_device.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.bf_device.device.destroy_image_view(*v, None));
        self.bf_device.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
pub struct AppData {
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    msaa_samples: vk::SampleCountFlags,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    command_pool: vk::CommandPool,
    // Color
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // Texture
    mip_levels: u32,
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    // Model
    indices: Vec<u32>,
    // Buffers
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

//=================================
// Physical Device
//=================================

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

//=======================
// Color Objects
//=======================

unsafe fn create_color_objects(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let (color_image, color_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,
        data.msaa_samples,
        data.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;

    data.color_image_view = create_image_view(
        device,
        data.color_image,
        data.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    Ok(())
}

//======================
// Texture
//======================

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let image = File::open("resources/viking_room.png")?;
    //let image = File::open("resources/orange-3d.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();

    if width != 1024 || height != 1024 || reader.info().color_type != png::ColorType::Rgba {
        panic!("Invalid texture image.");
    }

    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());

    device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED 
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
    )?;

    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    Ok(())
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        return Err(anyhow!("Texture image format does not support linear blitting!"));
    }

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        device.cmd_blit_image(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }

        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    barrier.subresource_range.base_mip_level = mip_levels - 1;
    barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
    barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
    barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn create_texture_image_view(device: &Device, data: &mut AppData) -> Result<()> {
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    Ok(())
}

unsafe fn create_texture_sampler(device: &Device, data: &mut AppData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.0)
        .max_lod(data.mip_levels as f32)
        .mip_lod_bias(0.0);

    data.texture_sampler = device.create_sampler(&info, None)?;

    Ok(())
}

//=====================
// Model
//=====================

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D { width, height, depth: 1 });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    _format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
    let (
        src_access_mask,
        dst_access_mask,
        src_stage_mask,
        dst_stage_mask,
    ) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => return Err(anyhow!("Unsupported image layout transition!")),
    };

    let command_buffer = begin_single_time_commands(device, data)?;

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn begin_single_time_commands(
    device: &Device,
    data: &AppData,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_commands(
    device: &Device,
    data: &AppData,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers);

    device.queue_submit(data.graphics_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D { 
            width, 
            height, 
            depth: 1 
        })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .samples(samples)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::ImageCreateFlags::empty());

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        device.update_descriptor_sets(
            &[ubo_write, sampler_write], 
            &[] as &[vk::CopyDescriptorSet],
        );
    }

    Ok(())
}

unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let size = (size_of::<u32>() * data.indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;

    memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::BufferCreateFlags::empty());

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &AppData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &AppData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = instance.get_physical_device_memory_properties(data.physical_device);

    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl Vertex {
    pub const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self { pos, color, tex_coord }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();

        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();

        [pos, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.color == other.color
            && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    view: Mat4,
    proj: Mat4,
}
