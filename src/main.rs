use macroquad::prelude::*;
use rodio::{buffer::SamplesBuffer, Decoder, OutputStream, OutputStreamHandle, Sink};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;
use midly::{Smf, MidiMessage, Timing, MetaMessage, TrackEventKind};
use ::rand::{thread_rng, Rng}; 

// --- 基础物理常数 (Mode A 使用) ---
const E_STEEL: f64 = 2.0e11;
const RHO_STEEL: f64 = 7850.0;
const RHO_COPPER: f64 = 8960.0; 
const TENSION_BASE: f64 = 670.0;
const HAMMER_MASS_BASE: f64 = 0.006;

// --- 仿真设置 ---
const FS_AUDIO: u32 = 44100;
const OVERSAMPLE: usize = 8; 
const FS_PHY: f64 = (FS_AUDIO as f64) * (OVERSAMPLE as f64);
const DT_PHY: f64 = 1.0 / FS_PHY;
const NOTE_DURATION: f64 = 4.0;

// --- 声音模式枚举 ---
#[derive(PartialEq, Clone, Copy)]
enum SoundMode {
    Physical,   // 物理建模
    Synthetic,  // 采样器
}

// --- 物理滤波器工具 (Mode A 使用) ---

#[derive(Clone)]
struct Biquad {
    b0: f64, b1: f64, b2: f64,
    a1: f64, a2: f64,
    z1: f64, z2: f64,
}

impl Biquad {
    fn bandpass(freq: f64, fs: f64, q: f64) -> Self {
        let omega = 2.0 * PI * freq / fs;
        let sn = omega.sin();
        let cs = omega.cos();
        let alpha = sn / (2.0 * q);

        let a0 = 1.0 + alpha;
        let b0 = alpha;   
        let b1 = 0.0;
        let b2 = -alpha;
        let a1 = -2.0 * cs;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
            z1: 0.0, z2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let out = self.b0 * input + self.z1;
        self.z1 = self.b1 * input + self.z2 - self.a1 * out;
        self.z2 = self.b2 * input - self.a2 * output(out);
        out
    }
}
fn output(o: f64) -> f64 { o }

// 声板链路 (Mode A)
#[derive(Clone)]
struct SoundboardChain {
    lp_y: f64, lp_alpha: f64,
    hp_x_prev: f64, hp_y_prev: f64, hp_alpha: f64,
    
    res1: Biquad,
    res2: Biquad,
    res3: Biquad,
}

impl SoundboardChain {
    fn new(hp_freq: f64, lp_freq: f64, fs: f64) -> Self {
        let dt = 1.0 / fs;
        let rc_lp = 1.0 / (2.0 * PI * lp_freq);
        let lp_alpha = dt / (rc_lp + dt);
        let rc_hp = 1.0 / (2.0 * PI * hp_freq);
        let hp_alpha = rc_hp / (rc_hp + dt);

        let res1 = Biquad::bandpass(500.0, fs, 3.0);
        let res2 = Biquad::bandpass(1200.0, fs, 2.0);
        let res3 = Biquad::bandpass(2800.0, fs, 1.2);

        Self { 
            lp_y: 0.0, lp_alpha,
            hp_x_prev: 0.0, hp_y_prev: 0.0, hp_alpha,
            res1, res2, res3
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let hp_out = self.hp_alpha * (self.hp_y_prev + input - self.hp_x_prev);
        self.hp_x_prev = input;
        self.hp_y_prev = hp_out;

        self.lp_y += self.lp_alpha * (hp_out - self.lp_y);
        let base_signal = self.lp_y;

        let r1 = self.res1.process(base_signal);
        let r2 = self.res2.process(base_signal);
        let r3 = self.res3.process(base_signal);

        base_signal + 0.12 * r1 + 0.08 * r2 + 0.04 * r3
    }
}

struct DcBlocker {
    x_prev: f32,
    y_prev: f32,
    r: f32,
}
impl DcBlocker {
    fn new() -> Self { Self { x_prev: 0.0, y_prev: 0.0, r: 0.995 } } 
    fn process(&mut self, input: f32) -> f32 {
        let output = input - self.x_prev + self.r * self.y_prev;
        self.x_prev = input;
        self.y_prev = output;
        output
    }
}

// --- 单根琴弦物理模型 (Mode A) ---
struct PianoString {
    y_curr: Vec<f64>,
    y_prev: Vec<f64>,
    y_next: Vec<f64>,
    c1: f64, c2: f64, c3: f64, c4: f64, c5: f64,
    dx: f64,
    hammer_pos: f64,
    hammer_vel: f64,
    hammer_mass: f64,
    hammer_k: f64, 
    hammer_p: f64, 
    is_hammer_active: bool,
    strike_idx: usize,
    bridge_idx: usize,
    
    hf_damping_coeff: f64, 
    f_hist1: f64, f_hist2: f64,
    
    bridge_diff1: f64, bridge_diff2: f64,
    bridge_vel: f64, bridge_disp: f64,
    
    start_delay: usize,
    step_count: usize,
    leak_loss: f64,
    subsonic_loss: f64,
    tilt_factor: f64,
    out_smooth: f64,
    
    noise_intensity: f64,
    sb_feedback: f64,
}

impl PianoString {
    fn new(midi_index: usize, velocity: f64, detune_factor: f64, seed_offset: f64) -> Self {
        let freq_target = 440.0 * 2.0f64.powf((midi_index as f64 - 69.0) / 12.0) * detune_factor;
        let note_ratio = (midi_index as f64 - 21.0).max(0.0) / (108.0 - 21.0);
        let is_bass = midi_index < 42; 

        let mut rng = thread_rng();
        let rand_var = 1.0 + (rng.gen::<f64>() - 0.5) * 0.05 * seed_offset;

        let core_radius = if is_bass { 0.00045 + note_ratio * 0.0001 } else { 0.0004 + note_ratio * 0.00015 };
        let outer_radius = if is_bass { 0.0022 - note_ratio * 0.0012 } else { core_radius };
        let area_core = PI * core_radius * core_radius;
        let area_outer = PI * outer_radius * outer_radius;
        let rho_l = if is_bass { RHO_STEEL * area_core + RHO_COPPER * (area_outer - area_core) } else { RHO_STEEL * area_core };
        
        let c_wave = (TENSION_BASE / rho_l).sqrt();
        let moment_inertia = PI * core_radius.powi(4) / 4.0;
        let kappa = (E_STEEL * moment_inertia) / rho_l;

        let mut length = c_wave / (2.0 * freq_target); 
        for _ in 0..15 { 
            let stiffness_term = (PI * PI * kappa) / (c_wave * c_wave * length * length);
            let f_actual = (c_wave / (2.0 * length)) * (1.0 + stiffness_term).sqrt();
            length = length * (f_actual / freq_target);
        }

        let b1 = (0.1 + note_ratio * 1.5) * rand_var; 
        let b2 = (if is_bass { 3.0e-4 } else { 2.5e-5 }) * rand_var;

        let hammer_k = 0.5e9 + note_ratio * 3.0e9; 
        let hammer_mass = HAMMER_MASS_BASE * (0.8 + (1.0 - note_ratio) * 3.0); 
        
        let vel_norm = (velocity / 127.0).clamp(0.0, 1.0);
        let hammer_p = 2.0 + 1.0 * vel_norm;
        let tilt_factor = 1.0 + 0.6 * vel_norm;
        let noise_intensity = 0.02 * vel_norm;

        let min_dx = (2.0 * kappa.sqrt() * DT_PHY).sqrt() * 3.0; 
        let nx = (length / min_dx).ceil() as usize;
        let safe_nx = if nx > 6 { nx } else { 6 };
        let dx = length / safe_nx as f64;

        let r = c_wave * DT_PHY / dx;
        let mu = (kappa * DT_PHY * DT_PHY) / dx.powi(4);
        let coeff_b1 = b1 * DT_PHY;
        let coeff_b2 = 2.0 * b2 * DT_PHY / dx.powi(2);
        let inv_denom = 1.0 / (1.0 + coeff_b1);

        let s_target = (0.13 * safe_nx as f64) as usize;
        let b_target = (safe_nx as f64 * 0.98) as usize;
        let strike_idx = s_target.clamp(2, safe_nx.saturating_sub(3));
        let bridge_idx = b_target.clamp(2, safe_nx.saturating_sub(3));

        let hf_damping_coeff = 0.002; 
        let leak_loss = if is_bass { 0.0001 } else { 0.00005 }; 
        let subsonic_loss = if is_bass { 0.00005 } else { 0.00002 }; 
        let start_delay = (rng.gen::<f64>() * 10.0 * seed_offset) as usize;

        Self {
            y_curr: vec![0.0; safe_nx],
            y_prev: vec![0.0; safe_nx],
            y_next: vec![0.0; safe_nx],
            c1: r*r, c2: mu, c3: coeff_b1, c4: coeff_b2, c5: inv_denom,
            dx,
            hammer_pos: 0.0,
            hammer_vel: velocity,
            hammer_mass,
            hammer_k,
            hammer_p,
            is_hammer_active: true,
            strike_idx,
            bridge_idx,
            f_hist1: 0.0, f_hist2: 0.0,
            bridge_vel: 0.0,
            bridge_disp: 0.0,
            hf_damping_coeff,
            start_delay,
            step_count: 0,
            leak_loss,
            subsonic_loss,
            tilt_factor,
            out_smooth: 0.0,
            bridge_diff1: 0.0,
            bridge_diff2: 0.0,
            noise_intensity,
            sb_feedback: 0.0,
        }
    }

    #[inline(always)]
    fn step(&mut self, noise_sample: f64) -> f64 {
        if self.step_count < self.start_delay {
            self.step_count += 1;
            return 0.0;
        }

        let nx = self.y_curr.len();
        let mut hammer_force_smoothed = 0.0;

        if self.is_hammer_active {
            let s_idx = self.strike_idx;
            let string_disp = unsafe { 
                0.25 * *self.y_curr.get_unchecked(s_idx - 1) +
                0.50 * *self.y_curr.get_unchecked(s_idx) +
                0.25 * *self.y_curr.get_unchecked(s_idx + 1)
            };
            
            let compression = self.hammer_pos - string_disp;
            let mut raw_force = 0.0;
            if compression > 0.0 {
                let mod_factor = 1.0 + noise_sample * self.noise_intensity;
                raw_force = self.hammer_k * compression.powf(self.hammer_p) * mod_factor;
            } else if self.hammer_pos < -0.002 { 
                self.is_hammer_active = false;
            }

            hammer_force_smoothed = 0.25 * raw_force + 0.5 * self.f_hist1 + 0.25 * self.f_hist2;
            self.f_hist2 = self.f_hist1;
            self.f_hist1 = raw_force;

            let accel = -hammer_force_smoothed / self.hammer_mass;
            self.hammer_vel += accel * DT_PHY;
            self.hammer_pos += self.hammer_vel * DT_PHY;
        } else {
            self.f_hist1 = 0.0; self.f_hist2 = 0.0;
        }

        unsafe {
            let y_curr_ptr = self.y_curr.as_ptr();
            let y_prev_ptr = self.y_prev.as_ptr();
            let y_next_ptr = self.y_next.as_mut_ptr();

            for i in 2..nx-2 {
                let yc = *y_curr_ptr.add(i);
                let yp = *y_prev_ptr.add(i);
                
                let dxx = *y_curr_ptr.add(i + 1) - 2.0 * yc + *y_curr_ptr.add(i - 1);
                let dxxxx = *y_curr_ptr.add(i + 2) - 4.0 * *y_curr_ptr.add(i + 1) + 6.0 * yc 
                          - 4.0 * *y_curr_ptr.add(i - 1) + *y_curr_ptr.add(i - 2);
                
                let v_xx = (*y_curr_ptr.add(i + 1) - 2.0 * yc + *y_curr_ptr.add(i - 1)) 
                         - (*y_prev_ptr.add(i + 1) - 2.0 * yp + *y_prev_ptr.add(i - 1));

                let mut force_term = 0.0;
                let s_idx = self.strike_idx;
                
                if i >= s_idx - 1 && i <= s_idx + 1 {
                    let mass_seg = RHO_STEEL * (PI * 0.0005 * 0.0005) * self.dx; 
                    let weight = if i == s_idx { 0.5 } else { 0.25 };
                    force_term = (hammer_force_smoothed * weight / mass_seg) * DT_PHY.powi(2);
                }

                let mut val = (2.0 * yc 
                                - yp * (1.0 - self.c3)
                                + self.c1 * dxx 
                                - self.c2 * dxxxx 
                                + self.c4 * v_xx 
                                + force_term) * self.c5;
                
                val -= self.hf_damping_coeff * dxxxx;
                val -= self.subsonic_loss * yc;
                
                if (i as isize - self.bridge_idx as isize).abs() <= 2 {
                    val -= self.leak_loss * yc;
                }

                *y_next_ptr.add(i) = val;
            }
        }

        self.y_prev.copy_from_slice(&self.y_curr);
        self.y_curr.copy_from_slice(&self.y_next);
        
        let b_idx = self.bridge_idx;
        let spatial_diff = self.y_curr[b_idx + 1] - self.y_curr[b_idx - 1]; 
        
        let feedback_force = 0.15 * self.sb_feedback;
        self.bridge_vel -= feedback_force;
        
        self.bridge_vel += spatial_diff * DT_PHY;
        self.bridge_disp += self.bridge_vel * DT_PHY;

        let raw_bridge = self.bridge_vel;
        self.bridge_diff1 = 0.85 * self.bridge_diff1 + 0.15 * raw_bridge;
        self.bridge_diff2 = 0.60 * self.bridge_diff2 + 0.40 * self.bridge_diff1;
        let diffused_bridge = 0.7 * self.bridge_diff2 + 0.3 * raw_bridge;

        let radiated = (0.9 * self.tilt_factor) * diffused_bridge + 0.1 * self.bridge_disp;
        
        self.out_smooth = 0.95 * self.out_smooth + 0.05 * radiated;
        
        self.out_smooth
    }
}

// --- 高级声音生成器 (Mode A) ---
struct AdvancedPianoKey {
    string_a: PianoString, 
    string_b: PianoString, 
    pan_l: f32, 
    pan_r: f32,
    dc_l: DcBlocker,
    dc_r: DcBlocker,
    sb_chain: SoundboardChain, 
    gain: f64,
    is_bass: bool,
    
    noise_gen: ::rand::rngs::ThreadRng,
    hf_env: f64,
}

impl AdvancedPianoKey {
    fn new(midi: usize) -> Self {
        let detune = 1.0002; 
        let pan_ratio = (midi as f32 - 21.0).max(0.0) / (108.0 - 21.0);
        let pi_f32 = PI as f32;
        let pan_l = ((1.0 - pan_ratio) * pi_f32 / 2.0).sin();
        let pan_r = (pan_ratio * pi_f32 / 2.0).sin();
        let is_bass = midi < 48;

        let f0 = 440.0 * 2.0f64.powf((midi as f64 - 69.0) / 12.0);
        let rad_comp = (f0 / 440.0).powf(0.8).clamp(0.4, 2.5);
        let gain = 6000000.0 / rad_comp;

        let sb_chain = SoundboardChain::new(20.0, 10000.0, FS_AUDIO as f64);

        Self {
            string_a: PianoString::new(midi, 5.0, 1.0, 1.0), 
            string_b: PianoString::new(midi, 5.0, detune, 2.0),
            pan_l,
            pan_r,
            dc_l: DcBlocker::new(),
            dc_r: DcBlocker::new(),
            sb_chain,
            gain,
            is_bass,
            noise_gen: thread_rng(),
            hf_env: 1.0,
        }
    }

    fn generate_stereo_sample(&mut self) -> (f32, f32) {
        let mut sum_a = 0.0;
        let mut sum_b = 0.0;

        for _ in 0..OVERSAMPLE {
            let n1 = self.noise_gen.gen::<f64>() - 0.5;
            let n2 = self.noise_gen.gen::<f64>() - 0.5;
            
            sum_a += self.string_a.step(n1);
            sum_b += self.string_b.step(n2);
        }

        let val_a = sum_a / OVERSAMPLE as f64;
        let val_b = sum_b / OVERSAMPLE as f64;

        let beat_gain = if self.is_bass { 0.6 } else { 0.2 };
        let raw_mix = (val_a + val_b) + beat_gain * (val_a - val_b).abs();
        
        let sb_out = self.sb_chain.process(raw_mix);

        let fb = sb_out * 0.05;
        self.string_a.sb_feedback = fb;
        self.string_b.sb_feedback = fb;

        if !self.is_bass {
            self.hf_env *= 0.9995;
        }
        let shaped = sb_out * (0.7 + 0.3 * self.hf_env);

        let amplified = shaped * self.gain;
        let clipped = amplified.clamp(-1.0, 1.0);

        let clean_l = self.dc_l.process(clipped as f32 * self.pan_l);
        let clean_r = self.dc_r.process(clipped as f32 * self.pan_r);
        
        (clean_l, clean_r)
    }
}


struct SoundBank { samples: HashMap<usize, Vec<f32>> }
impl SoundBank { fn new() -> Self { Self { samples: HashMap::new() } } }

struct AudioEngine {
    stream_handle: Option<OutputStreamHandle>,
    active_sinks: HashMap<usize, Sink>,
    enabled: bool,
}

impl AudioEngine {
    fn note_on(&mut self, bank: &SoundBank, midi: usize, velocity: f64, mode: SoundMode) {
        if !self.enabled { return; }
        if let Some(handle) = &self.stream_handle {
            if let Some(sink) = self.active_sinks.remove(&midi) { sink.stop(); }
            if let Ok(sink) = Sink::try_new(handle) {
                // 音源选择逻辑
                match mode {
                    SoundMode::Physical => {
                        if let Some(data) = bank.samples.get(&midi) {
                            let source = SamplesBuffer::new(2, FS_AUDIO, data.clone());
                            let vol = (velocity / 127.0).powf(1.5).max(0.01) as f32; 
                            sink.set_volume(vol);
                            sink.append(source);
                            self.active_sinks.insert(midi, sink);
                        }
                    },
                    SoundMode::Synthetic => {
                        // 采样器模式 (Sampled)
                        let path_str = get_sample_path(midi as u8); 
                        let path = Path::new(&path_str);
                        
                        if let Ok(file) = File::open(path) {
                            let reader = BufReader::new(file);
                            if let Ok(source) = Decoder::new(reader) {
                                let vol = (velocity / 127.0).powf(2.0).max(0.01) as f32;
                                sink.set_volume(vol);
                                sink.append(source);
                                // 采样通常是自然衰减，不需要 active_sinks 强行 stop (除非要断奏)
                                // 但为了统一管理 note_off，还是加进去
                                self.active_sinks.insert(midi, sink);
                            } else {
                                println!("Failed to decode MP3: {}", path_str);
                            }
                        } else {
                             // 如果找不到文件，就静默或打印错误
                             // println!("Sample not found: {}", path_str);
                        }
                    }
                }
            }
        }
    }
    fn note_off(&mut self, midi: usize) {
        if !self.enabled { return; }
        if let Some(sink) = self.active_sinks.remove(&midi) { sink.stop(); }
    }
}

// [新增] 采样路径生成器
fn get_sample_path(midi: u8) -> String {
    if midi < 21 || midi > 108 { return String::new(); }
    
    let index = midi - 20; 
    let octave = (midi / 12) - 1;
    let note_idx = (midi % 12) as usize;
    
    // 映射表: 0=C, 1=C#(cz), ...
    let notes = ["c", "cz", "d", "dz", "e", "f", "fz", "g", "gz", "a", "az", "b"];
    let name = notes[note_idx];
    

    format!(
        "{}/note/p_{:02}_{}{}.mp3",
        std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .display(),
        index,
        name,
        octave
    )
}

#[derive(Debug, Clone)]
struct MidiEvent { time_sec: f64, midi: u8, velocity: u8, is_on: bool }

enum AppState {
    LoadingSamples { current_midi: usize, target_midi: usize, start_midi: usize },
    Ready,
}

#[derive(Debug, Clone)]
enum SafeOwnedEvent {
    NoteOn { key: u8, vel: u8 },
    NoteOff { key: u8 },
    Tempo(u32),
    Ignore,
}

#[macroquad::main("Piano Midi Player")]
async fn main() {
    let audio_output = OutputStream::try_default();
    let _stream_guard;
    let mut audio_engine;

    match audio_output {
        Ok((stream, handle)) => {
            _stream_guard = Some(stream);
            audio_engine = AudioEngine { stream_handle: Some(handle), active_sinks: HashMap::new(), enabled: true };
        },
        Err(_) => {
            _stream_guard = None;
            audio_engine = AudioEngine { stream_handle: None, active_sinks: HashMap::new(), enabled: false };
        }
    }

    // [范围调整] C2=36, C7=96
    let midi_start = 36; 
    let midi_end = 96;  
    let mut sound_bank = SoundBank::new();
    let mut app_state = AppState::LoadingSamples { 
        current_midi: midi_start, target_midi: midi_end, start_midi: midi_start 
    };

    let white_w = 26.0;
    let white_h = 160.0;
    let black_w = 16.0;
    let black_h = 100.0;
    
    // [显示优化] 自动计算缩放比例以适应 C2-C7
    // C2-C7 约 36 个白键
    let total_white_keys = 36.0;
    let needed_width = total_white_keys * white_w;
    // 留出左右边距 20px
    let mut scale_factor = 1.0; 

    let mut midi_score: Vec<MidiEvent> = Vec::new();
    let mut is_playing_midi = false;
    let mut midi_start_time = 0.0;
    let mut midi_play_head_idx = 0;
    let mut current_filename = String::from("None");
    let mut loading_status = String::from("");
    let mut mouse_holding_midi: Option<usize> = None;
    
    let mut sound_mode = SoundMode::Physical;

    // [按键映射] 完整覆盖 C2-C7 白键
    let mut key_map: Vec<(usize, KeyCode, char)> = vec![];
    
    // Row 1: 1-0 -> C2-E3
    let row1 = [
        (36, KeyCode::Key1, '1'), (38, KeyCode::Key2, '2'), (40, KeyCode::Key3, '3'), (41, KeyCode::Key4, '4'),
        (43, KeyCode::Key5, '5'), (45, KeyCode::Key6, '6'), (47, KeyCode::Key7, '7'), (48, KeyCode::Key8, '8'),
        (50, KeyCode::Key9, '9'), (52, KeyCode::Key0, '0')
    ];
    
    // Row 2: Q-P -> F3-A4
    let row2 = [
        (53, KeyCode::Q, 'Q'), (55, KeyCode::W, 'W'), (57, KeyCode::E, 'E'), (59, KeyCode::R, 'R'),
        (60, KeyCode::T, 'T'), (62, KeyCode::Y, 'Y'), (64, KeyCode::U, 'U'), (65, KeyCode::I, 'I'),
        (67, KeyCode::O, 'O'), (69, KeyCode::P, 'P')
    ];

    // Row 3: A-L -> B4-C6
    let row3 = [
        (71, KeyCode::A, 'A'), (72, KeyCode::S, 'S'), (74, KeyCode::D, 'D'), (76, KeyCode::F, 'F'),
        (77, KeyCode::G, 'G'), (79, KeyCode::H, 'H'), (81, KeyCode::J, 'J'), (83, KeyCode::K, 'K'),
        (84, KeyCode::L, 'L')
    ];

    // Row 4: Z-M -> D6-C7
    let row4 = [
        (86, KeyCode::Z, 'Z'), (88, KeyCode::X, 'X'), (89, KeyCode::C, 'C'), (91, KeyCode::V, 'V'),
        (93, KeyCode::B, 'B'), (95, KeyCode::N, 'N'), (96, KeyCode::M, 'M')
    ];

    for k in row1.iter() { key_map.push(*k); }
    for k in row2.iter() { key_map.push(*k); }
    for k in row3.iter() { key_map.push(*k); }
    for k in row4.iter() { key_map.push(*k); }
    
    // 用于跟踪哪些键正在被按下
    let mut active_key_notes: HashMap<KeyCode, usize> = HashMap::new();

    loop {
        clear_background(color_u8!(30, 32, 40, 255));

        match app_state {
            AppState::LoadingSamples { ref mut current_midi, target_midi, start_midi } => {
                if *current_midi <= target_midi {
                    let total_frames = (NOTE_DURATION * FS_AUDIO as f64) as usize;
                    let mut buffer = Vec::with_capacity(total_frames * 2);
                    let mut key_sim = AdvancedPianoKey::new(*current_midi);
                    
                    for _ in 0..total_frames { 
                        let (l, r) = key_sim.generate_stereo_sample();
                        buffer.push(l);
                        buffer.push(r);
                    }
                    sound_bank.samples.insert(*current_midi, buffer);
                    
                    let progress = (*current_midi - start_midi) as f32 / (target_midi - start_midi) as f32;
                    let cx = screen_width() / 2.0;
                    let cy = screen_height() / 2.0;
                    draw_text("Pre-baking Physics ...", cx - 200.0, cy - 40.0, 30.0, WHITE);
                    draw_rectangle(cx - 200.0, cy + 20.0, 400.0, 20.0, DARKGRAY);
                    draw_rectangle(cx - 200.0, cy + 20.0, 400.0 * progress, 20.0, SKYBLUE);
                    *current_midi += 1;
                } else {
                    app_state = AppState::Ready;
                }
            }
            AppState::Ready => {
                let time_now = get_time();

                // 自动缩放计算
                // C2-C7 约 36 个白键
                let total_white_keys = 36.0;
                let needed_width = total_white_keys * white_w;
                // 留出左右边距 20px
                scale_factor = (screen_width() - 40.0) / needed_width;
                if scale_factor > 1.2 { scale_factor = 1.2; } // 限制最大放大

                let start_x = (screen_width() - (needed_width * scale_factor)) / 2.0;
                let scaled_white_w = white_w * scale_factor;
                let scaled_white_h = white_h * scale_factor;
                let scaled_black_w = black_w * scale_factor;
                let scaled_black_h = black_h * scale_factor;
                let keys_start_y = screen_height() - scaled_white_h - 20.0;

                draw_rectangle(0.0, 0.0, screen_width(), 90.0, color_u8!(45, 45, 50, 255));
                draw_text("Piano Sim", 20.0, 30.0, 30.0, WHITE);
                
                if ui_button(20.0, 45.0, 160.0, 30.0, "Open .mid File") {
                    if let Some(path) = rfd::FileDialog::new().add_filter("MIDI", &["mid", "midi"]).pick_file() {
                        current_filename = path.file_name().unwrap().to_string_lossy().to_string();
                        if let Ok(bytes) = fs::read(&path) {
                            midi_score = parse_midi_file(&bytes);
                            loading_status = format!("Loaded {} events", midi_score.len());
                            is_playing_midi = false;
                            midi_play_head_idx = 0;
                            for i in 0..128 { audio_engine.note_off(i); }
                        } else {
                            loading_status = String::from("Failed to read file");
                        }
                    }
                }

                // Mode Button
                let mode_label = if sound_mode == SoundMode::Physical { "Mode: Physical (A)" } else { "Mode: Sampler (B)" };
                let mode_color = if sound_mode == SoundMode::Physical { SKYBLUE } else { ORANGE };
                if ui_button_colored(200.0, 45.0, 180.0, 30.0, mode_label, mode_color) {
                    sound_mode = match sound_mode {
                        SoundMode::Physical => SoundMode::Synthetic,
                        SoundMode::Synthetic => SoundMode::Physical,
                    };
                }
                
                draw_text(&format!("File: {}", current_filename), 400.0, 55.0, 20.0, LIGHTGRAY);
                draw_text(&loading_status, 400.0, 75.0, 16.0, GREEN);

                let play_label = if is_playing_midi { "STOP" } else { "PLAY" };
                let btn_color = if is_playing_midi { RED } else { GREEN };
                if ui_button_colored(screen_width() - 120.0, 45.0, 100.0, 30.0, play_label, btn_color) {
                    if is_playing_midi {
                        is_playing_midi = false;
                        for i in 0..128 { audio_engine.note_off(i); }
                    } else if !midi_score.is_empty() {
                        is_playing_midi = true;
                        midi_start_time = time_now;
                        midi_play_head_idx = 0;
                    }
                }

                if is_playing_midi {
                    let elapsed = time_now - midi_start_time;
                    while midi_play_head_idx < midi_score.len() {
                        let ev = &midi_score[midi_play_head_idx];
                        if elapsed >= ev.time_sec {
                            let midi = ev.midi as usize;
                            if midi >= midi_start && midi <= midi_end {
                                if ev.is_on { 
                                    audio_engine.note_on(&sound_bank, midi, ev.velocity as f64, sound_mode); 
                                } else { 
                                    audio_engine.note_off(midi); 
                                }
                            }
                            midi_play_head_idx += 1;
                        } else { break; }
                    }
                    if midi_play_head_idx >= midi_score.len() { is_playing_midi = false; }
                }

                // --- 键盘输入逻辑 (Shift 处理) ---
                let shift_down = is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift);
                
                for (base_midi, kcode, _) in &key_map {
                    // 1. 按下处理
                    if is_key_pressed(*kcode) {
                        // 判断是否要升半音 (Shift + White = Black)
                        // 黑键逻辑：如果 base_midi + 1 是黑键，则触发 midi+1
                        // C(0), C#(1), D(2), D#(3), E(4), F(5), F#(6), G(7), G#(8), A(9), A#(10), B(11)
                        // 黑键 index: 1, 3, 6, 8, 10
                        let next_midi = base_midi + 1;
                        let next_is_black = matches!(next_midi % 12, 1 | 3 | 6 | 8 | 10);
                        
                        let target_midi = if shift_down && next_is_black {
                            next_midi
                        } else {
                            *base_midi
                        };
                        
                        audio_engine.note_on(&sound_bank, target_midi, 90.0, sound_mode);
                        active_key_notes.insert(*kcode, target_midi);
                    }
                    
                    // 2. 松开处理
                    if is_key_released(*kcode) {
                        // 停止该按键之前触发的那个音
                        if let Some(played_midi) = active_key_notes.remove(kcode) {
                            audio_engine.note_off(played_midi);
                        }
                    }
                }

                // --- 绘制键盘 ---
                let mut x_pos = start_x;
                let mut black_keys = Vec::new();
                let mouse_pos = mouse_position();
                let mouse_down = is_mouse_button_down(MouseButton::Left);
                let mouse_released = is_mouse_button_released(MouseButton::Left);
                let mut hovered_midi: Option<usize> = None;

                let mut current_midi = midi_start;
                while current_midi <= midi_end {
                    let is_white = is_white_key(current_midi);
                    if is_white {
                        let rect = Rect::new(x_pos, keys_start_y, scaled_white_w, scaled_white_h);
                        // 检查播放状态 (音频引擎或键盘按住)
                        let is_playing = audio_engine.active_sinks.contains_key(&current_midi);
                        // 检查物理按键状态
                        let is_phys_pressed = active_key_notes.values().any(|&m| m == current_midi);
                        
                        let color = if is_playing || is_phys_pressed { color_u8!(100, 200, 255, 255) } else { WHITE };
                        
                        draw_rectangle(rect.x + 2.0, rect.y + 2.0, rect.w, rect.h, Color::new(0.0,0.0,0.0,0.2));
                        draw_rectangle(rect.x, rect.y, rect.w, rect.h, color);
                        draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 2.0, color_u8!(200,200,200,255));
                        
                        if rect.contains(vec2(mouse_pos.0, mouse_pos.1)) && mouse_pos.1 > keys_start_y + scaled_black_h {
                             hovered_midi = Some(current_midi);
                        }
                        
                        // 显示按键字符
                        if let Some((_, _, ch)) = key_map.iter().find(|(m, _, _)| *m == current_midi) {
                             draw_text(&ch.to_string(), rect.x + rect.w/2.0 - 5.0, rect.y + rect.h - 10.0, 16.0, BLACK);
                        } else if current_midi % 12 == 0 {
                            draw_text(&format!("C{}", (current_midi/12)-1), rect.x+4.0, rect.y+rect.h-30.0, 12.0, GRAY);
                        }
                        x_pos += scaled_white_w;
                    } else {
                        // 黑键位置: 在上一个白键右边减去一半黑键宽
                        let b_rect = Rect::new(x_pos - (scaled_black_w / 2.0), keys_start_y, scaled_black_w, scaled_black_h);
                        black_keys.push((current_midi, b_rect));
                    }
                    current_midi += 1;
                }

                for (midi, rect) in black_keys {
                    let is_playing = audio_engine.active_sinks.contains_key(&midi);
                    let is_phys_pressed = active_key_notes.values().any(|&m| m == midi);
                    
                    let color = if is_playing || is_phys_pressed { color_u8!(50, 100, 200, 255) } else { BLACK };
                    draw_rectangle(rect.x + 2.0, rect.y + 2.0, rect.w, rect.h, Color::new(0.0,0.0,0.0,0.5));
                    draw_rectangle(rect.x, rect.y, rect.w, rect.h, color);
                    if !is_playing && !is_phys_pressed {
                        draw_rectangle(rect.x + 2.0, rect.y + rect.h - 10.0, rect.w - 4.0, 5.0, color_u8!(60,60,60,255));
                    }
                    
                    // 黑键不显示字符，因为是 Shift+白键
                    
                    if rect.contains(vec2(mouse_pos.0, mouse_pos.1)) { hovered_midi = Some(midi); }
                }

                if mouse_down {
                    if let Some(midi) = hovered_midi {
                        if mouse_holding_midi != Some(midi) {
                            if let Some(old) = mouse_holding_midi { audio_engine.note_off(old); }
                            audio_engine.note_on(&sound_bank, midi, 90.0, sound_mode);
                            mouse_holding_midi = Some(midi);
                        }
                    } else {
                         if let Some(old) = mouse_holding_midi { audio_engine.note_off(old); mouse_holding_midi = None; }
                    }
                }
                if mouse_released {
                    if let Some(midi) = mouse_holding_midi { audio_engine.note_off(midi); }
                    mouse_holding_midi = None;
                }
            }
        }
        next_frame().await
    }
}

// --- 辅助函数 ---

fn is_white_key(midi: usize) -> bool {
    matches!(midi % 12, 0 | 2 | 4 | 5 | 7 | 9 | 11)
}

fn ui_button(x: f32, y: f32, w: f32, h: f32, label: &str) -> bool {
    ui_button_colored(x, y, w, h, label, DARKGRAY)
}

fn ui_button_colored(x: f32, y: f32, w: f32, h: f32, label: &str, color: Color) -> bool {
    let mouse_pos = mouse_position();
    let rect = Rect::new(x, y, w, h);
    let is_hover = rect.contains(vec2(mouse_pos.0, mouse_pos.1));
    let is_click = is_hover && is_mouse_button_pressed(MouseButton::Left);
    let draw_color = if is_hover { Color::new(color.r + 0.1, color.g + 0.1, color.b + 0.1, 1.0) } else { color };
    draw_rectangle(rect.x, rect.y, rect.w, rect.h, draw_color);
    draw_rectangle_lines(rect.x, rect.y, rect.w, rect.h, 2.0, WHITE);
    draw_text(label, rect.x + 10.0, rect.y + 20.0, 20.0, WHITE);
    is_click
}

fn parse_midi_file(bytes: &[u8]) -> Vec<MidiEvent> {
    let smf = match Smf::parse(bytes) { Ok(s) => s, Err(_) => return Vec::new() };
    let mut events = Vec::new();
    let mut ppq = 480.0;
    if let Timing::Metrical(t) = smf.header.timing { ppq = t.as_int() as f64; }
    let mut tempo_us = 500_000.0; 
    struct RawEvent { abs_tick: u64, kind: SafeOwnedEvent }
    let mut raw_events = Vec::new();
    for track in smf.tracks.into_iter() {
        let mut abs_tick = 0;
        for event in track.into_iter() {
            abs_tick += event.delta.as_int() as u64;
            let kind = match event.kind {
                TrackEventKind::Midi { message, .. } => {
                    match message {
                        MidiMessage::NoteOn { key, vel } => SafeOwnedEvent::NoteOn { key: key.as_int(), vel: vel.as_int() },
                        MidiMessage::NoteOff { key, .. } => SafeOwnedEvent::NoteOff { key: key.as_int() },
                        _ => SafeOwnedEvent::Ignore,
                    }
                },
                TrackEventKind::Meta(MetaMessage::Tempo(t)) => SafeOwnedEvent::Tempo(t.as_int()),
                _ => SafeOwnedEvent::Ignore,
            };
            if let SafeOwnedEvent::Ignore = kind { continue; }
            raw_events.push(RawEvent { abs_tick, kind });
        }
    }
    raw_events.sort_by_key(|e| e.abs_tick);
    let mut current_time = 0.0;
    let mut last_tick = 0;
    for event in raw_events {
        let delta_tick = event.abs_tick - last_tick;
        current_time += (delta_tick as f64) * (tempo_us / 1_000_000.0) / ppq;
        last_tick = event.abs_tick;
        match event.kind {
            SafeOwnedEvent::Tempo(us) => tempo_us = us as f64,
            SafeOwnedEvent::NoteOn { key, vel } => {
                if vel > 0 {
                    events.push(MidiEvent { time_sec: current_time, midi: key, velocity: vel, is_on: true });
                } else {
                    events.push(MidiEvent { time_sec: current_time, midi: key, velocity: 0, is_on: false });
                }
            },
            SafeOwnedEvent::NoteOff { key } => {
                events.push(MidiEvent { time_sec: current_time, midi: key, velocity: 0, is_on: false });
            },
            _ => {}
        }
    }
    events
}