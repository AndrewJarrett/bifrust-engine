#![allow(
    dead_code,
    unused_variables,
    unused_imports,
    clippy::too_many_argumnets,
    clippy::unnecessary_wraps
)]

use anyhow::{anyhow, Result, Error};
use log::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::Version;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Bifrost Engine Tester")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };

    event_loop.set_control_flow(ControlFlow::Poll);

    let _ = event_loop.run(move |event, elwt| {
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                unsafe { app.render(&window) }.unwrap();
            }
            // Destroy our Vulkan app
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
                unsafe { app.destroy(); }
            }
            _ => {}
        }
    });

    return Ok(());
}

    /*
    let surface = unsafe {
        PresentationSurface::from_window(window, &adapter.physical_device)
            .unwrap()
    };


    let mut swapchain_config = window.get_preferred_swap_chain_config(&adapter.physical_device);
    swapchain_config.present_mode = gfx_hal::window::PresentMode::Fifo;
    let (mut swapchain, backbuffer) = unsafe {
        device.create_swapchain(&mut surface, swapchain_config, None)
            .unwrap()
    };
    */

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let instance = Self::create_instance(window, &entry)?;
        Ok(Self { entry, instance })
    }

    /// Create our Vulkan instance.
    unsafe fn create_instance(window: &Window, entry: &Entry) -> Result<Instance> {
        let application_info = vk::ApplicationInfo::builder()
            .application_name(b"Bifrost Engine\0")
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(b"No Engine\0")
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        let mut extensions = vk_window::get_required_instance_extensions(window)
            .iter()
            .map(|e| e.as_ptr())
            .collect::<Vec<_>>();

        // Required by Vulkan SDK on macOS since 1.3.216
        let flags = if
            cfg!(target_os = "macos") &&
            entry.version()? >= PORTABILITY_MACOS_VERSION 
        {
                info!("Enabling extensions for macOS portability.");
                extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
                extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

        let info = vk::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_extension_names(&extensions)
            .flags(flags);

        Ok(entry.create_instance(&info, None)?)
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys our Vulkan app.
    unsafe fn destroy(&mut self) {
        self.instance.destroy_instance(None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {}
