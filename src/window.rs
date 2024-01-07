#[allow(
    dead_code
)]

use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder}
};
use std::io::Result;

#[derive(Debug)]
pub struct BfWindow {
    pub window: Window,
    pub event_loop: EventLoop<()>,
    pub destroying: bool,
    pub minimized: bool,
}

impl BfWindow {

    pub fn new(width: u16, height: u16, title: String) -> Result<Self> {
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width, height))
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


        Ok(Self {
            window,
            event_loop,
            destroying: false,
            minimized: false,
        })
    }

}
