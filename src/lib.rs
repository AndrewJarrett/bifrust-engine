#![allow(
    unused_imports
)]

use winit::{
    event::*,
    dpi::LogicalSize,
    event_loop::{ControlFlow, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
    window::{WindowBuilder, CursorGrabMode},
};
use vulkanalia::prelude::v1_0::*;
use log::info;

#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use crate::app::App;

pub mod app;

#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub fn run() {
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
    let window = WindowBuilder::new()
        .with_title("Bifrust Engine Tester")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("app")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    // Grab the cursor
    /*
    window.set_cursor_grab(CursorGrabMode::Confined)
        .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
        .unwrap();
    */

    // App
    let mut app = unsafe { App::create(&window).unwrap() };

    let extension_count = unsafe { app.get_extension_count() };
    info!("Found {} physical device extension!", extension_count);

    let mut close_requested = false;
    let mut destroying = false;
    let mut minimized = false;

    event_loop.set_control_flow(ControlFlow::Poll);

    let _ = event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw to render continuously
            Event::AboutToWait if !destroying && !minimized => {
                if close_requested {
                    elwt.exit();
                } else {
                    window.request_redraw();
                }
            }
            // Render a frame if our Vulkan app is not being destroyed.
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                app.delta_time = app.start.elapsed().as_secs_f32();

                unsafe { app.render(&window) }.unwrap();
            }
            // Handle user input - keyboard
            Event::WindowEvent { event: WindowEvent::KeyboardInput { event, .. }, .. } => {
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
                            close_requested = true;
                        }
                        _ => { }
                    }
                }
            }
            // Mouse movement
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                app.delta_mouse = delta;
                //dbg!(app.delta_mouse);
            }
            // Handle window is resized
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            }
            // Destroy our Vulkan app
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                destroying = true;
                elwt.exit();
                unsafe { app.device.device_wait_idle().unwrap(); }
                unsafe { app.destroy(); }
            }
            _ => {}
        }
    });
}
