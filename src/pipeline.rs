#![allow(
    dead_code,
    unused_variables
)]

use std::io::Result;

pub struct Pipeline {
    vert: Vec<u8>,
    frag: Vec<u8>,
}

impl Pipeline {

    pub fn new(vert_file_path: String, frag_file_path: String) -> Result<Self> {
        let vert = include_bytes!("../shaders/vert.spv").to_vec();
        let frag = include_bytes!("../shaders/frag.spv").to_vec();

        Ok(Self {
            vert,
            frag,
        })
    }
}
