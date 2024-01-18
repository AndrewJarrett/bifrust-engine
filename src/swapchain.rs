use crate::window::BfWindow;
use crate::device::{BfDevice, QueueFamilyIndices, SwapchainSupport};

use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtension;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub sync: Sync,
    pub frame: usize,
}

impl Swapchain {

    pub unsafe fn new(
        bf_window: &BfWindow,
        bf_device: &BfDevice,
    ) -> Result<Self> {
        println!("Swapchain new()");
        let indices = QueueFamilyIndices::get(&bf_device.instance, &bf_device.physical_device, &bf_device.surface)?;
        println!("QFI after");
        let support = SwapchainSupport::get(&bf_device.instance, &bf_device.physical_device, &bf_device.surface)?;
        println!("Swapchainsupport after");

        let surface_format = Self::get_swapchain_surface_format(&support.formats);
        println!("get surf format");
        let present_mode = Self::get_swapchain_present_mode(&support.present_modes);
        println!("get present mode");
        let extent = Self::get_swapchain_extent(&bf_window, support.capabilities);
        println!("get extent");

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(bf_device.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());
        println!("swapchain info khr");

        let format = surface_format.format;
        let swapchain = bf_device.device.create_swapchain_khr(&info, None)?;
        println!("create swapchain");
        let images = bf_device.device.get_swapchain_images_khr(swapchain)?;
        println!("get swap images");

        let image_views = Self::create_swapchain_image_views(&bf_device, &images, format)?;
        println!("get image views");

        let render_pass = Self::create_render_pass(&bf_device, format)?;
        println!("render pass");

        let (depth_image, depth_image_memory, depth_image_view) = Self::create_depth_objects(
            &bf_device,
            extent,
        )?;

        let framebuffers = Self::create_framebuffers(&bf_device, &image_views, /*color_image_view, */depth_image_view, render_pass, extent)?;

        let sync = Self::create_sync_objects(&bf_device, &images)?;

        Ok(Self {
            swapchain,
            format,
            extent,
            images,
            image_views,
            render_pass,
            depth_image,
            depth_image_memory,
            depth_image_view,
            framebuffers,
            sync,
            frame: 0,
        })
    }

    fn get_swapchain_surface_format(
        formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| formats[0])
    }

    fn get_swapchain_present_mode(
        present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        present_modes
            .iter()
            .cloned()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    }

    fn get_swapchain_extent(bf_window: &BfWindow, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            let size = bf_window.window.inner_size();
            let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
            vk::Extent2D::builder()
                .width(clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                    size.width,
                ))
                .height(clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                    size.height,
                ))
                .build()
        }
    }

    pub fn get_extent_aspect_ratio(&self) -> Result<f32> {
        Ok(self.extent.width as f32 / self.extent.height as f32)
    }

    pub fn acquire_next_image(&self, bf_device: &BfDevice) -> Result<VkSuccessResult<u32>> {
        let in_flight_fence = self.sync.in_flight_fences[self.frame];

        let result: VkSuccessResult<u32>;
        unsafe {
            bf_device.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

            result = bf_device.device.acquire_next_image_khr(
                self.swapchain,
                u64::MAX,
                self.sync.image_available_semaphores[self.frame],
                vk::Fence::null(),
            );
        }

        Ok(result)
    }

    pub unsafe fn submit_command_buffers(
        &mut self,
        bf_device: &BfDevice,
        image_index: usize
    ) -> Result<VkResult<vk::SuccessCode>> {
        let image_in_flight = self.sync.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            bf_device.device.wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.sync.images_in_flight[image_index] = image_in_flight;

        //self.update_command_buffer(image_index)?;
        //self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.sync.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[bf_device.command_buffers[image_index]];
        let signal_semaphores = &[self.sync.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        let in_flight_fence = self.sync.in_flight_fences[self.frame];
        bf_device.device.reset_fences(&[in_flight_fence])?;

        bf_device.device
            .queue_submit(bf_device.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = bf_device.device.queue_present_khr(bf_device.present_queue, &present_info);

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(result)
    }

    unsafe fn create_swapchain_image_views(
        bf_device: &BfDevice,
        images: &Vec<vk::Image>,
        format: vk::Format
    ) -> Result<Vec<vk::ImageView>> {
        let image_views = images
            .iter()
            .map(|i| Self::create_image_view(&bf_device, *i, format, vk::ImageAspectFlags::COLOR, 1))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(image_views)
    }

    unsafe fn create_image_view(
        bf_device: &BfDevice,
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

        Ok(bf_device.device.create_image_view(&info, None)?)
    }

    unsafe fn create_render_pass(bf_device: &BfDevice, format: vk::Format) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(bf_device.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_stencil_attachment = vk::AttachmentDescription::builder()
            .format(Self::get_depth_format(&bf_device)?)
            .samples(bf_device.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_resolve_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_resolve_attachment_ref = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let resolve_attachments = &[color_resolve_attachment_ref];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(resolve_attachments);

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        let attachments = &[
            color_attachment,
            depth_stencil_attachment,
            color_resolve_attachment
        ];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        Ok(bf_device.device.create_render_pass(&info, None)?)
    }

    unsafe fn create_depth_objects(
        bf_device: &BfDevice,
        extent: vk::Extent2D,
    ) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let format = Self::get_depth_format(&bf_device)?;
        
        let (depth_image, depth_image_memory) = Self::create_image(
            &bf_device,
            extent.width,
            extent.height,
            1,
            bf_device.msaa_samples,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let depth_image_view = Self::create_image_view(
            &bf_device, 
            depth_image, 
            format,
            vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        Ok((depth_image, depth_image_memory, depth_image_view))
    }

    unsafe fn get_depth_format(bf_device: &BfDevice) -> Result<vk::Format> {
        let candidates = &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        Self::get_supported_format(
            bf_device,
            candidates,
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }

    unsafe fn get_supported_format(
        bf_device: &BfDevice,
        candidates: &[vk::Format],
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format> {
        candidates
            .iter()
            .cloned()
            .find(|f| {
                let properties = bf_device.instance.get_physical_device_format_properties(
                    bf_device.physical_device,
                    *f,
                );

                match tiling {
                    vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                    vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                    _ => false,
                }

            })
            .ok_or_else(|| anyhow!("Failed to find supported format!"))
    }

    unsafe fn create_image(
        bf_device: &BfDevice,
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

        let image = bf_device.device.create_image(&info, None)?;

        let requirements = bf_device.device.get_image_memory_requirements(image);

        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(Self::get_memory_type_index(
                &bf_device,
                properties,
                requirements,
            )?);

        let image_memory = bf_device.device.allocate_memory(&info, None)?;

        bf_device.device.bind_image_memory(image, image_memory, 0)?;

        Ok((image, image_memory))
    }

    unsafe fn get_memory_type_index(
        bf_device: &BfDevice,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32> {
        let memory = bf_device.instance.get_physical_device_memory_properties(bf_device.physical_device);

        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
    }

    unsafe fn create_framebuffers(
        bf_device: &BfDevice,
        image_views: &Vec<vk::ImageView>,
        //color_image_view: vk::ImageView,
        depth_image_view: vk::ImageView,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>> {
        let framebuffers = image_views
            .iter()
            .map(|i| {
                let attachments = &[/*color_image_view, */depth_image_view, *i];
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass)
                    .attachments(attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                bf_device.device.create_framebuffer(&create_info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(framebuffers)
    }

    unsafe fn create_sync_objects(
        bf_device: &BfDevice,
        images: &Vec<vk::Image>,
    ) -> Result<Sync> {
        let mut sync = Sync::new()?;

        let semaphore_info = vk::SemaphoreCreateInfo::builder();

        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            sync.image_available_semaphores
                .push(bf_device.device.create_semaphore(&semaphore_info, None)?);
            sync.render_finished_semaphores
                .push(bf_device.device.create_semaphore(&semaphore_info, None)?);

            sync.in_flight_fences.push(bf_device.device.create_fence(&fence_info, None)?);
        }

        sync.images_in_flight = images
            .iter()
            .map(|_| vk::Fence::null())
            .collect();

        Ok(sync)
    }

}

#[derive(Debug)]
pub struct Sync {
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

impl Sync {

    pub unsafe fn new() -> Result<Self> {
        let image_available_semaphores: Vec<vk::Semaphore> = Vec::new();
        let render_finished_semaphores: Vec<vk::Semaphore> = Vec::new();
        let in_flight_fences: Vec<vk::Fence> = Vec::new();
        let images_in_flight: Vec<vk::Fence> = Vec::new();

        Ok(Self {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        })
    }

}
