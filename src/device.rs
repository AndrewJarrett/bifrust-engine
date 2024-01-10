#![allow(
    dead_code,
)]

use crate::window::BfWindow;

use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use anyhow::{anyhow, Result};
use log::*;
use thiserror::Error;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::Version;

use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::ExtDebugUtilsExtension;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

#[derive(Debug)]
pub struct BfDevice {
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub device: Device,
    pub surface: vk::SurfaceKHR,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub properties: vk::PhysicalDeviceProperties,
    pub msaa_samples: vk::SampleCountFlags,
    instance: Instance,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: vk::PhysicalDevice,
}

impl BfDevice {

    pub unsafe fn new(bf_window: &BfWindow) -> Result<Self> {
        let entry = Self::create_entry()?;
        let (instance, messenger) = Self::create_instance(&bf_window, &entry)?;

        let surface = Self::create_surface(&bf_window, &instance)?;
        let (properties, physical_device) = Self::pick_physical_device(&instance, surface)?;
        let msaa_samples = Self::get_max_msaa_samples(properties);

        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &entry,
            &instance,
            physical_device,
            surface
        )?;

        let (command_pool, command_pools) = Self::create_command_pools(
            &instance,
            physical_device,
            surface,
            &device,
            2
        )?;
        
        Ok(Self {
            command_pool,
            command_pools,
            device,
            surface,
            graphics_queue,
            present_queue,
            properties,
            msaa_samples,
            instance,
            messenger,
            physical_device,
        })
    }

    unsafe fn create_entry() -> Result<Entry> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;

        Ok(entry)
    }

    unsafe fn create_instance(bf_window: &BfWindow, entry: &Entry)
        -> Result<(Instance, Option<vk::DebugUtilsMessengerEXT>)> {
        // Application Info
        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"Bifrust Engine\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        // Layers
        let available_layers = entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>();

        if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
            return Err(anyhow!("Validation layer requested but not supported."));
        }

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            Vec::new()
        };

        // Extensions
        let mut extensions = vk_window::get_required_instance_extensions(&bf_window.window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        // Required by Vulkan SDK on macOS since 1.3.216
        let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for macOS portability.");
            extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        if VALIDATION_ENABLED {
            extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        }

        // Create
        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(flags);

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .user_callback(Some(debug_callback));

        if VALIDATION_ENABLED {
            info = info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&info, None)?;

        // Messenger
        let mut messenger = None;
        if VALIDATION_ENABLED {
            messenger = Some(instance.create_debug_utils_messenger_ext(&debug_info, None)?);
        }

        Ok((instance, messenger))
    }

    fn create_surface(bf_window: &BfWindow, instance: &Instance) -> Result<vk::SurfaceKHR> {
        let surface = bf_window.create_surface(instance)?;

        Ok(surface)
    }

    unsafe fn pick_physical_device(instance: &Instance, surface: vk::SurfaceKHR) -> Result<(vk::PhysicalDeviceProperties, vk::PhysicalDevice)> {
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);

            info!("Max push constants is {}.", properties.limits.max_push_constants_size);

            if let Err(error) = Self::check_physical_device(&instance, physical_device, surface) {
                warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
            } else {
                info!("Selected physical device (`{}`).", properties.device_name);
                return Ok((properties, physical_device));
            }
        }

        Err(anyhow!("Failed to find suitable physical device."))
    }

    unsafe fn check_physical_device(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR
    ) -> Result<()> {
        QueueFamilyIndices::get(&instance, physical_device, surface)?;
        Self::check_physical_device_extensions(&instance, physical_device)?;

        let support = SwapchainSupport::get(&instance, physical_device, surface)?;
        if support.formats.is_empty() || support.present_modes.is_empty() {
            return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
        }

        let features = instance.get_physical_device_features(physical_device);
        if features.sampler_anisotropy != vk::TRUE {
            return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
        }

        Ok(())
    }

    unsafe fn check_physical_device_extensions(
        instance: &Instance,
        physical_device: vk::PhysicalDevice
    ) -> Result<()> {
        let extensions = instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>();
        if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
            Ok(())
        } else {
            Err(anyhow!(SuitabilityError("Missing required device extensions.")))
        }
    }

    unsafe fn get_max_msaa_samples(
        properties: vk::PhysicalDeviceProperties
    ) -> vk::SampleCountFlags {
        let counts = properties.limits.framebuffer_color_sample_counts
            & properties.limits.framebuffer_depth_sample_counts;
        [
            vk::SampleCountFlags::_64,
            vk::SampleCountFlags::_32,
            vk::SampleCountFlags::_16,
            vk::SampleCountFlags::_8,
            vk::SampleCountFlags::_4,
            vk::SampleCountFlags::_2,
        ]
        .iter()
        .cloned()
        .find(|c| counts.contains(*c))
        .unwrap_or(vk::SampleCountFlags::_1)
    }

    unsafe fn create_logical_device(
        entry: &Entry,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<(Device, vk::Queue, vk::Queue)> {
        let indices = QueueFamilyIndices::get(&instance, physical_device, surface)?;

        let mut unique_indices = HashSet::new();
        unique_indices.insert(indices.graphics);
        unique_indices.insert(indices.present);

        let queue_priorities = &[1.0];
        let queue_infos = unique_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let layers = if VALIDATION_ENABLED {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let mut extensions = DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();

        // Required by Vulkan SDK on macOS since 1.3.216
        if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .sample_rate_shading(true);

        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = instance.create_device(physical_device, &info, None)?;

        let graphics_queue = device.get_device_queue(indices.graphics, 0);
        let present_queue = device.get_device_queue(indices.present, 0);

        Ok((device, graphics_queue, present_queue))
    }

    unsafe fn create_command_pools(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        device: &Device,
        num_images: usize,
    ) -> Result<(vk::CommandPool, Vec<vk::CommandPool>)> {
        let command_pool = Self::create_command_pool(&instance, physical_device, surface, &device)?;
        let mut command_pools: Vec<vk::CommandPool> = Vec::new();

        for _ in 0..num_images {
            let command_pool = Self::create_command_pool(instance, physical_device, surface, &device)?;
            command_pools.push(command_pool);
        }

        Ok((command_pool, command_pools))
    }

    unsafe fn create_command_pool(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        device: &Device,
    ) -> Result<vk::CommandPool> {
        let indices = QueueFamilyIndices::get(&instance, physical_device, surface)?;

        let info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(indices.graphics);

        Ok(device.create_command_pool(&info, None)?)
    }

    unsafe fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .flags(vk::BufferCreateFlags::empty());

        let buffer = self.device.create_buffer(&buffer_info, None)?;

        let requirements = self.device.get_buffer_memory_requirements(buffer);

        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(self.get_memory_type_index(properties, requirements)?);

        let buffer_memory = self.device.allocate_memory(&memory_info, None)?;

        self.device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        Ok((buffer, buffer_memory))
    }

    pub unsafe fn begin_single_time_commands(&self) -> Result<vk::CommandBuffer> {
        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = self.device.allocate_command_buffers(&info)?[0];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        Ok(command_buffer)
    }

    pub unsafe fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        self.device.end_command_buffer(command_buffer)?;

        let command_buffers = &[command_buffer];
        let info = vk::SubmitInfo::builder()
            .command_buffers(command_buffers);

        self.device.queue_submit(self.graphics_queue, &[info], vk::Fence::null())?;
        self.device.queue_wait_idle(self.graphics_queue)?;

        self.device.free_command_buffers(self.command_pool, &[command_buffer]);

        Ok(())
    }

    pub unsafe fn copy_buffer(
        &self,
        source: vk::Buffer,
        destination: vk::Buffer,
        size: vk::DeviceSize,
    ) -> Result<()> {
        let command_buffer = self.begin_single_time_commands()?;

        let regions = vk::BufferCopy::builder().size(size);
        self.device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

        self.end_single_time_commands(command_buffer)?;

        Ok(())
    }

    pub unsafe fn copy_buffer_to_image(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let command_buffer = self.begin_single_time_commands()?;

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

        self.device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        self.end_single_time_commands(command_buffer)?;

        Ok(())
    }

    pub unsafe fn create_image(
        &self,
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

        let image = self.device.create_image(&info, None)?;

        let requirements = self.device.get_image_memory_requirements(image);

        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(self.get_memory_type_index(properties, requirements)?);

        let image_memory = self.device.allocate_memory(&info, None)?;

        self.device.bind_image_memory(image, image_memory, 0)?;

        Ok((image, image_memory))
    }

    unsafe fn get_memory_type_index(
        &self,
        properties: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<u32> {
        let memory = self.instance.get_physical_device_memory_properties(self.physical_device);

        (0..memory.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let memory_type = memory.memory_types[*i as usize];
                suitable && memory_type.property_flags.contains(properties)
            })
            .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
    }

    pub unsafe fn destroy(&self) -> Result<()> {
        self.command_pools.iter().for_each(|p| self.device.destroy_command_pool(*p, None));
        self.device.destroy_command_pool(self.command_pool, None);
        self.device.destroy_device(None);

        unsafe { self.instance.destroy_surface_khr(self.surface, None); };

        if VALIDATION_ENABLED && self.messenger.is_some() {
            self.instance.destroy_debug_utils_messenger_ext(self.messenger.unwrap(), None);
        }

        self.instance.destroy_instance(None);

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance.get_physical_device_surface_capabilities_khr(physical_device, surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, surface)?,
            present_modes: instance.get_physical_device_surface_present_modes_khr(physical_device, surface)?,
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    graphics: u32,
    present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let properties = instance
            .get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, _properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                surface,
            )? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

