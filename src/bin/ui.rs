use lenia::{Cycle, PackageLenia, Shape};
use eframe::egui::{self, Color32, Frame, Key, Pos2, RichText, Stroke, Ui, UiBuilder, Vec2};
use std::fs;
use std::path::Path;
use std::process::{Command, Child};
use std::net::TcpStream;
use std::io::{Write, Read};
use std::time::Instant;

struct Handler {
    arrayfire: Child,
    stream: TcpStream,
    buffer: [u8;1024],
    timers: Vec<Instant>,
    pull_lenia: bool,
    push_lenia: bool,
    load_lenia: (bool, u8),
    lenia: PackageLenia,
    delta: usize,
    kernel_shape: [f32;100],
    growth_shape: [f32;100],
    layer_nr: usize,
    channel_nr: usize,
    arrow: Arrow
}
pub fn draw_lines_in_box(ui: &mut Ui, values: &[f32], desired_width: f32, desired_height: f32, center: bool) {
    let rect = ui.allocate_space(Vec2::new(desired_width, desired_height)).1;
    let ui_builder = UiBuilder::new().max_rect(rect);
    let child_ui = &mut ui.new_child(ui_builder);

    Frame::default()
        .stroke(Stroke::new(1.0, Color32::GRAY))
        .show(child_ui, |ui| {
            let painter = ui.painter();
            let available_width = ui.available_width();
            let available_height = ui.available_height();

            let num_values = values.len();
            if num_values == 0 {return}

            let x_step = available_width / (num_values as f32 - 1.0).max(1.0);
            let y_scale = available_height;

            for (x_index, &y_value) in values.iter().enumerate() {
                let x_pos = x_step * x_index as f32;
                let y_pos = y_scale * y_value;

                let y = if center {(rect.max.y + rect.min.y)/2.} else {rect.max.y};

                let start_pos = Pos2::new(x_pos + rect.min.x, y);
                let end_pos = Pos2::new(x_pos + rect.min.x, y - y_pos);

                painter.line_segment(
                    [start_pos, end_pos],
                    Stroke::new(1.0, Color32::DARK_GRAY),
                );
            }
        }
    );
}

struct Arrow {
    user: usize,
    runner: usize
}
impl Default for Arrow {
    fn default() -> Self {
        Self { user: 0, runner: 0 }
    }
}
impl Arrow {
    fn cursor_f32(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context, x: &mut f32) {
        self.runner += 1;
        if self.user != self.runner - 1 {return}
        ui.label(RichText::new("^^^^^^^^").color(Color32::RED));
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {*x -= 0.001}
        else if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {*x += 0.001}
    }
    fn cursor_usize(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context, x: &mut usize) {
        self.runner += 1;
        if self.user != self.runner - 1 { return }
        ui.label(RichText::new("^^^^^^^^").color(Color32::RED));
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) { *x -= 1 }
        else if ctx.input(|i| i.key_pressed(Key::ArrowRight)) { *x += 1 }
    }
    fn cursor_bool(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context, x: &mut bool) {
        self.runner += 1;
        if self.user != self.runner - 1 { return }
        ui.label(RichText::new("^^^^^^^^").color(Color32::RED));
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::Enter)) 
            { *x = !(*x) }
    }
    fn cursor_shape(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context, x: &mut Shape) {
        self.runner += 1;
        if self.user != self.runner - 1 {return}
        ui.label(RichText::new("^^^^^^^^").color(Color32::RED));
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft)) {x.previous()}
        else if ctx.input(|i| i.key_pressed(Key::ArrowRight)) {x.next()}
    }
    fn cursor_lenia(&mut self, ui: &mut Ui, ctx: &eframe::egui::Context) -> bool {
        self.runner += 1;
        if self.user != self.runner - 1 {return false}
        ui.label(RichText::new("^^^^^^^^").color(Color32::RED));
        if ctx.input(|i| i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::Enter)) 
            { return true }
        false
    }
}

impl eframe::App for Handler {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
            if ctx.input(|i| i.key_pressed(Key::ArrowUp)) {self.arrow.user -= 1}
            if ctx.input(|i| i.key_pressed(Key::ArrowDown)) {self.arrow.user += 1}
            if ctx.input(|i| i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::Enter)) 
                { self.push_lenia = true }
            if ctx.input(|i| i.key_pressed(Key::Q)) {
                self.arrayfire.kill().unwrap();
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }

            ui.label(&format!("{}ms",self.delta));
            ui.heading("LeniaUI");
            ui.heading("<<<<<<>>>>>>");

            // TODO, remember what number have lenia at the moment
            let dirs = fs::read_dir("data/").unwrap().filter(|e| e.is_ok() ).filter(|e| e.as_ref().unwrap().path().is_dir() )
                .map(|e| e.unwrap().path().file_name().unwrap().to_str().unwrap().to_owned().trim().parse::<usize>() ).filter(|e| e.is_ok() )
                .map(|e| e.unwrap() ).collect::<Vec<usize>>();
            ui.label(format!("Found presets:"));
            dirs.iter().for_each(|d| { 
                ui.label(format!(" - {}", d));
                if self.load_lenia.0 {self.arrow.cursor_lenia(ui, ctx);}
                else {self.load_lenia = (self.arrow.cursor_lenia(ui, ctx), (*d) as u8)}
            });

            ui.heading("<<<<<<>>>>>>");
            ui.label(format!("Layer nr: {}", self.layer_nr));
            self.arrow.cursor_usize(ui, ctx, &mut self.layer_nr);

            if self.lenia.layers.len() > self.layer_nr {
                let layer = &mut self.lenia.layers[self.layer_nr];
                ui.label( format!("Layer key: {}", layer.0) );
                ui.label( format!("Layer source: {}", layer.1.source_key) );
                ui.heading("------------");
                ui.label( format!("Kernel") );
                ui.label( format!("Shape: {}", layer.1.kernel.shape) );
                self.arrow.cursor_shape(ui, ctx, &mut layer.1.kernel.shape );

                if layer.1.kernel.centering {ui.label( format!("With centering") );}
                else                        {ui.label( format!("Without centering") );}
                self.arrow.cursor_bool(ui, ctx, &mut layer.1.kernel.centering );
                if layer.1.kernel.hard_clip {ui.label( format!("With hard-clip") );}
                else                        {ui.label( format!("With sigmoid-clip") );}
                self.arrow.cursor_bool(ui, ctx, &mut layer.1.kernel.hard_clip );
                layer.1.kernel.parameters.iter_mut().enumerate().for_each(|(i,p)|{
                    match i % 3 {
                        0 => { ui.label(format!("{} Width: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p ); }
                        1 => { ui.label(format!("{} Offset: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p ); }
                        2 => { ui.label(format!("{} Strength: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p ); }
                        _ => {}
                    }
                });
                self.kernel_shape.iter_mut().enumerate().for_each(|(i,y)|{
                    let x = if i < 50 { 50. - i as f32 } else {(i+1) as f32 -50.};
                    *y = layer.1.kernel._calc((x+1.) /50.) 
                });
                draw_lines_in_box(ui, &self.kernel_shape, 200., 50., false);

                ui.heading("------------");

                ui.label( format!("Growth map") );
                ui.label( format!("Shape: {}", layer.1.growth_map.shape) );
                self.arrow.cursor_shape(ui, ctx, &mut layer.1.growth_map.shape );
                if layer.1.growth_map.centering {ui.label( format!("With centering") );}
                else                            {ui.label( format!("Without centering") );}
                self.arrow.cursor_bool(ui, ctx, &mut layer.1.growth_map.centering );
                if layer.1.growth_map.hard_clip {ui.label( format!("With hard-clip") );}
                else                            {ui.label( format!("With sigmoid-clip") );}
                self.arrow.cursor_bool(ui, ctx, &mut layer.1.growth_map.hard_clip );
                layer.1.growth_map.parameters.iter_mut().enumerate().for_each(|(i,p)|{
                    match i % 3 {
                        0 => { ui.label(format!("{} Width: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p );}
                        1 => { ui.label(format!("{} Offset: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p );}
                        2 => { ui.label(format!("{} Strength: {:>.4}", i, *p));
                            self.arrow.cursor_f32(ui, ctx, p );}
                        _ => {}
                    }
                });
                self.growth_shape.iter_mut().enumerate().for_each(|(x,y)| 
                    *y = layer.1.growth_map._calc((x+1) as f32 /100.) );
                draw_lines_in_box(ui, &self.growth_shape, 200., 50., true);
            }

            ui.heading("<<<<<<>>>>>>");
            ui.label(format!("Channel nr: {}", self.channel_nr));
            self.arrow.cursor_usize(ui, ctx, &mut self.channel_nr);
            if self.lenia.channels.len() > self.channel_nr {
                let channel = &mut self.lenia.channels[self.channel_nr];
                ui.label( format!("Channel key: {}", channel.0) );
                ui.label( format!("Destinations: ") );
                channel.1.keys.iter_mut().zip(channel.1.floats.iter_mut()).for_each(|(k,f)| {
                    ui.label( format!("Key: {} | Weight: {}", *k, *f) );
                    self.arrow.cursor_f32(ui, ctx, f);
                });
            }
            self.arrow.runner = 0;

            self.communicate(ctx);
        });
    }
}

impl Default for Handler {
    fn default() -> Self {
        let arrayfire = launch();
        let mut stream = None;
        while stream.is_none() {stream = connect();}

        Self { 
            arrayfire: arrayfire.unwrap(), 
            stream: stream.unwrap(), 
            buffer: [0;1024], 
            timers: vec![Instant::now(); 2],
            pull_lenia: true,
            push_lenia: true,
            load_lenia: (false, 0),
            lenia: PackageLenia::empty(),
            delta: 0,
            kernel_shape: [0.;100],
            growth_shape: [0.;100],
            layer_nr: 0,
            channel_nr: 0,
            arrow: Arrow::default()
        }
    }
}

impl Handler {

    fn communicate(&mut self, ctx: &eframe::egui::Context) -> bool {

        let request = 
        if ctx.input(|i| i.key_pressed(Key::P)) {vec![10]}
        else if ctx.input(|i| i.key_pressed(Key::S)) {vec![11]}
        else if ctx.input(|i| i.key_pressed(Key::N)) {vec![13]}
        else {vec![]};

        if !request.is_empty() { self.pull_lenia = true; self.send(&request); }
        
        if self.load_lenia.0 {
            self.send(&vec![12, self.load_lenia.1]);
            self.load_lenia.0 = false;
            self.pull_lenia = true;
        }
        if self.timers[0].elapsed().as_millis() > 250 { 
            let _ = self.send(&vec![9]);
            self.delta = self.buffer[0] as usize;
            self.timers[0] = Instant::now();
        }
        if self.pull_lenia {
            let r = self.send(&vec![7]);
            self.lenia = bincode::deserialize(&self.buffer[0..r]).unwrap();
            self.pull_lenia = false;
        }
        if self.push_lenia {
            let mut p = bincode::serialize(&self.lenia).unwrap();
            p.insert(0, 8);
            let _ = self.send(&p);
            self.push_lenia = false;
        }
        true
    }

    fn send(&mut self, request: &Vec<u8>) -> usize{
        match self.stream.write_all(request) {   
            Ok(_) => {
                match self.stream.read(&mut self.buffer) {
                    Ok(bytes_read) => {
                        bytes_read
                    }
                    Err(e) => {eprintln!("Error reading response: {}", e); 0}
                }
            }
            Err(e) => {eprintln!("Error sending request: {}", e); 0}
        }
    }

}



fn main() -> eframe::Result {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([512.0, 1024.0]),
        ..Default::default()
    };

    eframe::run_native(
        "LeniaUI",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::<Handler>::default())
        }),
    )
}


fn connect() -> Option<TcpStream> {
    match TcpStream::connect("127.0.0.1:2137") {
        Ok(stream) => Some(stream),
        Err(_) => None,
    }
}

fn launch() -> Option<Child> {
    let mut arrayfire: Option<Child> = None;
    
    let arrayfire_app_dir = Path::new("./");

    if let Ok(process) = Command::new("cargo")
        .args(&["run", "-r", "--bin", "compute"])
        .current_dir(arrayfire_app_dir)
        .spawn() {
        arrayfire = Some(process);

    } else {eprintln!("xd");}
    arrayfire
}

