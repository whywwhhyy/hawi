use chrono::{DateTime, Utc};
use serde_json::{json, Map, Value};
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

const MIN_CONTENT_WIDTH: f64 = 640.0;
const MIN_CONTENT_HEIGHT: f64 = 660.0;
const VERSION: &str = "hawi.core.v1";
const TYPE_JSON_FRAME: u8 = 0x01;
const DEFAULT_MAX_FRAME_SIZE: usize = 16 * 1024 * 1024;
const GRACEFUL_SHUTDOWN_TIMEOUT_MS: u64 = 800;
const DEFAULT_COMMAND_TIMEOUT_MS: u64 = 15_000;
const SESSION_COMMAND_TIMEOUT_MS: u64 = 30_000;
const FORK_COMMAND_TIMEOUT_MS: u64 = 60_000;
const COMPACT_COMMAND_TIMEOUT_MS: u64 = 120_000;
const MAX_LOADED_SESSIONS: usize = 5;

static CORE_INSTANCE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct EngineLauncher {
    command: String,
    args_prefix: Vec<String>,
    source: EngineLauncherSource,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineLauncherSource {
    Bundled,
    Uv,
}

#[derive(Clone)]
struct EnvPaths {
    repo_root: PathBuf,
    workspace_root: PathBuf,
    config_path: PathBuf,
    backend_log_path: PathBuf,
    engine_launcher: EngineLauncher,
}

struct GuiState {
    env: EnvPaths,
    inspect: Mutex<Value>,
    config: Mutex<Value>,
    manager: Arc<Mutex<SessionEngineManager>>,
    refreshed_providers: Mutex<HashSet<String>>,
}

struct CoreStartOptions {
    initial_session_id: Option<String>,
    initial_session_name: Option<String>,
    launch_profile: Option<Value>,
}

#[derive(Debug, Clone)]
struct CoreCommandError {
    message: String,
    details: Option<Value>,
}

impl CoreCommandError {
    fn message(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            details: None,
        }
    }

    fn from_error_frame(frame: Value) -> Self {
        let payload = frame_payload(&frame);
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Core error")
            .to_string();
        Self {
            message,
            details: payload.get("details").cloned(),
        }
    }
}

type CoreCommandResult = Result<Value, CoreCommandError>;
type CoreEmit = Arc<dyn Fn(String, Value) + Send + Sync + 'static>;

struct ManagerEngineEvent {
    session_id: String,
    channel: String,
    payload: Value,
}

struct CoreProcess {
    child: Option<Arc<Mutex<Child>>>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    pending: Arc<Mutex<HashMap<String, mpsc::Sender<CoreCommandResult>>>>,
    sequence: u64,
    instance_id: u64,
    emit: CoreEmit,
    repo_root: PathBuf,
    workspace_root: PathBuf,
    backend_log_path: PathBuf,
    engine_launcher: EngineLauncher,
}

impl CoreProcess {
    fn new(
        emit: CoreEmit,
        repo_root: PathBuf,
        workspace_root: PathBuf,
        backend_log_path: PathBuf,
        engine_launcher: EngineLauncher,
    ) -> Self {
        Self {
            child: None,
            stdin: None,
            pending: Arc::new(Mutex::new(HashMap::new())),
            sequence: 0,
            instance_id: CORE_INSTANCE_SEQUENCE.fetch_add(1, Ordering::SeqCst) + 1,
            emit,
            repo_root,
            workspace_root,
            backend_log_path,
            engine_launcher,
        }
    }

    fn is_running(&self) -> bool {
        let Some(child) = &self.child else {
            return false;
        };
        let Ok(mut child) = child.lock() else {
            return false;
        };
        matches!(child.try_wait(), Ok(None))
    }

    fn start(
        &mut self,
        next_config: &Value,
        metadata: &Value,
        refreshed_providers: &HashSet<String>,
        options: CoreStartOptions,
    ) -> Result<Value, String> {
        if value_string(next_config, "modelName").is_empty() {
            return Ok(json!({}));
        }
        self.stop("start-replace-existing");
        if let Some(parent) = self.backend_log_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create log directory: {error}"))?;
        }
        fs::write(&self.backend_log_path, "")
            .map_err(|error| format!("failed to clear backend log: {error}"))?;

        let plugin_config_path = env::temp_dir().join(format!(
            "hawi-gui-plugins-{}-{}.json",
            std::process::id(),
            self.instance_id
        ));
        fs::write(
            &plugin_config_path,
            serde_json::to_vec_pretty(
                next_config
                    .get("pluginConfigs")
                    .unwrap_or(&Value::Object(Map::new())),
            )
            .map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("failed to write plugin config: {error}"))?;

        let mut engine_args = Vec::new();
        engine_args.extend([
            "--model".to_string(),
            value_string(next_config, "modelName"),
        ]);
        for provider in refreshed_providers {
            engine_args.extend(["--refresh-provider".to_string(), provider.clone()]);
        }
        engine_args.extend(["--transport".to_string(), "stdio".to_string()]);
        engine_args.extend([
            "--system-prompt".to_string(),
            non_empty_string(next_config.get("systemPrompt"))
                .unwrap_or_else(|| value_string(metadata, "default_system_prompt")),
        ]);
        engine_args.extend([
            "--plugins".to_string(),
            string_list(next_config.get("selectedPlugins")).join(","),
        ]);
        engine_args.extend([
            "--plugin-config".to_string(),
            plugin_config_path.to_string_lossy().into_owned(),
        ]);
        if let Some(profile) = options.launch_profile {
            engine_args.extend(["--gui-launch-profile".to_string(), profile.to_string()]);
        }
        if let Some(session_id) = options.initial_session_id {
            engine_args.extend(["--initial-session-id".to_string(), session_id]);
        }
        if let Some(session_name) = options.initial_session_name {
            engine_args.extend(["--initial-session-name".to_string(), session_name]);
        }
        engine_args.extend(tool_call_purpose_engine_args(
            next_config
                .get("toolCallPurposeEnabled")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        ));
        engine_args.extend([
            "--log-file".to_string(),
            self.backend_log_path.to_string_lossy().into_owned(),
        ]);

        let args = build_engine_run_args(&self.repo_root, engine_args, &self.engine_launcher);
        let mut command = Command::new(&self.engine_launcher.command);
        command
            .args(&args)
            .current_dir(&self.workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .envs(build_engine_env(&self.repo_root, &self.engine_launcher));
        let mut child = command.spawn().map_err(|error| {
            format!("failed to spawn {}: {error}", self.engine_launcher.command)
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or("hawi-engine stdout was not piped")?;
        let stderr = child
            .stderr
            .take()
            .ok_or("hawi-engine stderr was not piped")?;
        let stdin = child
            .stdin
            .take()
            .ok_or("hawi-engine stdin was not piped")?;
        let child = Arc::new(Mutex::new(child));
        let stdin = Arc::new(Mutex::new(stdin));

        self.child = Some(child.clone());
        self.stdin = Some(stdin);
        self.pending.lock().expect("pending lock").clear();

        spawn_stdout_reader(stdout, self.pending.clone(), self.emit.clone());
        spawn_stderr_reader(stderr, self.emit.clone());
        spawn_exit_watcher(child, self.pending.clone(), self.emit.clone());

        let public_args = if self.engine_launcher.source == EngineLauncherSource::Uv {
            args.iter().skip(1).cloned().collect::<Vec<_>>()
        } else {
            args.clone()
        };
        Ok(json!({
            "command": self.engine_launcher.command,
            "args": public_args,
            "cwd": self.workspace_root,
            "engineSource": match self.engine_launcher.source {
                EngineLauncherSource::Bundled => "bundled",
                EngineLauncherSource::Uv => "uv",
            },
            "logFile": self.backend_log_path,
        }))
    }

    fn start_readonly(&mut self) -> Result<Value, String> {
        self.stop("readonly-start-replace-existing");
        if let Some(parent) = self.backend_log_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create log directory: {error}"))?;
        }
        fs::write(&self.backend_log_path, "")
            .map_err(|error| format!("failed to clear backend log: {error}"))?;

        let args = build_engine_run_args(
            &self.repo_root,
            vec![
                "--readonly".to_string(),
                "--transport".to_string(),
                "stdio".to_string(),
                "--log-file".to_string(),
                self.backend_log_path.to_string_lossy().into_owned(),
            ],
            &self.engine_launcher,
        );
        let mut command = Command::new(&self.engine_launcher.command);
        command
            .args(&args)
            .current_dir(&self.workspace_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .envs(build_engine_env(&self.repo_root, &self.engine_launcher));
        let mut child = command.spawn().map_err(|error| {
            format!("failed to spawn {}: {error}", self.engine_launcher.command)
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or("hawi-engine stdout was not piped")?;
        let stderr = child
            .stderr
            .take()
            .ok_or("hawi-engine stderr was not piped")?;
        let stdin = child
            .stdin
            .take()
            .ok_or("hawi-engine stdin was not piped")?;
        let child = Arc::new(Mutex::new(child));
        let stdin = Arc::new(Mutex::new(stdin));

        self.child = Some(child.clone());
        self.stdin = Some(stdin);
        self.pending.lock().expect("pending lock").clear();

        spawn_stdout_reader(stdout, self.pending.clone(), self.emit.clone());
        spawn_stderr_reader(stderr, self.emit.clone());
        spawn_exit_watcher(child, self.pending.clone(), self.emit.clone());

        let public_args = if self.engine_launcher.source == EngineLauncherSource::Uv {
            args.iter().skip(1).cloned().collect::<Vec<_>>()
        } else {
            args.clone()
        };
        Ok(json!({
            "command": self.engine_launcher.command,
            "args": public_args,
            "cwd": self.workspace_root,
            "engineSource": match self.engine_launcher.source {
                EngineLauncherSource::Bundled => "bundled",
                EngineLauncherSource::Uv => "uv",
            },
            "logFile": self.backend_log_path,
            "mode": "readonly",
        }))
    }

    fn stop(&mut self, reason: &str) {
        let shutdown_id = self.next_id();
        if let Some(stdin) = self.stdin.as_ref().cloned() {
            let frame = json!({
                "version": VERSION,
                "type": "shutdown",
                "id": shutdown_id,
                "payload": { "reason": reason },
            });
            let _ = stdin
                .lock()
                .map(|mut input| input.write_all(&encode_json_frame(&frame)));
        }

        let child = self.child.take();
        self.stdin = None;
        reject_all_pending(
            &self.pending,
            CoreCommandError::message("hawi-engine was stopped"),
        );

        let Some(child) = child else {
            return;
        };
        let deadline = now_ms() + GRACEFUL_SHUTDOWN_TIMEOUT_MS;
        while now_ms() < deadline {
            let exited = child
                .lock()
                .ok()
                .and_then(|mut child| child.try_wait().ok())
                .flatten()
                .is_some();
            if exited {
                return;
            }
            thread::sleep(Duration::from_millis(25));
        }
        {
            if let Ok(mut child) = child.lock() {
                let _ = child.kill();
            };
        }
    }

    fn send_command(
        &mut self,
        command_type: &str,
        payload: Value,
        timeout_ms: u64,
    ) -> CoreCommandResult {
        let stdin = self
            .stdin
            .as_ref()
            .ok_or_else(|| CoreCommandError::message("hawi-engine is not running"))?
            .clone();
        let id = self.next_id();
        let frame = json!({
            "version": VERSION,
            "type": command_type,
            "id": id,
            "payload": payload,
        });
        let (sender, receiver) = mpsc::channel();
        self.pending
            .lock()
            .expect("pending lock")
            .insert(id.clone(), sender);
        if let Err(error) = stdin
            .lock()
            .map_err(|_| CoreCommandError::message("failed to lock hawi-engine stdin"))?
            .write_all(&encode_json_frame(&frame))
        {
            self.pending.lock().expect("pending lock").remove(&id);
            return Err(CoreCommandError::message(format!(
                "failed to write core command: {error}"
            )));
        }
        match receiver.recv_timeout(Duration::from_millis(timeout_ms)) {
            Ok(result) => result,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.pending.lock().expect("pending lock").remove(&id);
                Err(CoreCommandError::message(format!(
                    "Core command timed out: {command_type}"
                )))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.pending.lock().expect("pending lock").remove(&id);
                Err(CoreCommandError::message(
                    "hawi-engine response channel closed",
                ))
            }
        }
    }

    fn next_id(&mut self) -> String {
        self.sequence += 1;
        format!("gui-{}-{}", base36(now_ms()), self.sequence)
    }
}

fn spawn_stdout_reader(
    mut stdout: impl Read + Send + 'static,
    pending: Arc<Mutex<HashMap<String, mpsc::Sender<CoreCommandResult>>>>,
    emit: CoreEmit,
) {
    thread::spawn(move || {
        let mut decoder = TlvDecoder::new(DEFAULT_MAX_FRAME_SIZE);
        let mut buffer = [0_u8; 8192];
        loop {
            match stdout.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => match decoder.push(&buffer[..count]) {
                    Ok(frames) => {
                        for (type_byte, value) in frames {
                            if type_byte != TYPE_JSON_FRAME {
                                continue;
                            }
                            let parsed = match serde_json::from_slice::<Value>(&value) {
                                Ok(frame) => frame,
                                Err(error) => {
                                    emit_bad_frame(&emit, error.to_string());
                                    continue;
                                }
                            };
                            if !is_core_frame(&parsed) {
                                emit_bad_frame(&emit, format!("Invalid core frame: {parsed}"));
                                continue;
                            }
                            if let Some(id) = frame_id(&parsed) {
                                if is_command_response_frame(&parsed) {
                                    let sender = pending.lock().expect("pending lock").remove(&id);
                                    if let Some(sender) = sender {
                                        let result = if frame_type(&parsed) == Some("error") {
                                            Err(CoreCommandError::from_error_frame(parsed))
                                        } else {
                                            Ok(parsed)
                                        };
                                        let _ = sender.send(result);
                                    }
                                    continue;
                                }
                            }
                            emit("core:event".to_string(), parsed);
                        }
                    }
                    Err(error) => emit_bad_frame(&emit, error),
                },
                Err(error) => {
                    emit_bad_frame(&emit, error.to_string());
                    break;
                }
            }
        }
    });
}

fn spawn_stderr_reader(stderr: impl Read + Send + 'static, emit: CoreEmit) {
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(line) => emit(
                    "core:stderr".to_string(),
                    Value::String(format!("{line}\n")),
                ),
                Err(error) => {
                    emit(
                        "core:stderr".to_string(),
                        Value::String(format!("{error}\n")),
                    );
                    break;
                }
            }
        }
    });
}

fn spawn_exit_watcher(
    child: Arc<Mutex<Child>>,
    pending: Arc<Mutex<HashMap<String, mpsc::Sender<CoreCommandResult>>>>,
    emit: CoreEmit,
) {
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(200));
        let status = child
            .lock()
            .ok()
            .and_then(|mut child| child.try_wait().ok())
            .flatten();
        if let Some(status) = status {
            reject_all_pending(
                &pending,
                CoreCommandError::message(format!("hawi-engine exited ({:?})", status.code())),
            );
            emit(
                "core:exit".to_string(),
                json!({
                    "code": status.code(),
                    "signal": Value::Null,
                }),
            );
            break;
        }
    });
}

fn emit_bad_frame(emit: &CoreEmit, message: String) {
    emit(
        "core:event".to_string(),
        json!({
            "version": VERSION,
            "type": "error",
            "payload": {
                "ok": false,
                "code": "bad_frame",
                "message": message,
            },
        }),
    );
}

fn reject_all_pending(
    pending: &Arc<Mutex<HashMap<String, mpsc::Sender<CoreCommandResult>>>>,
    error: CoreCommandError,
) {
    let waiters = pending
        .lock()
        .expect("pending lock")
        .drain()
        .map(|(_, sender)| sender)
        .collect::<Vec<_>>();
    for sender in waiters {
        let _ = sender.send(Err(error.clone()));
    }
}

struct TlvDecoder {
    buffer: Vec<u8>,
    max_frame_size: usize,
}

impl TlvDecoder {
    fn new(max_frame_size: usize) -> Self {
        Self {
            buffer: Vec::new(),
            max_frame_size,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Result<Vec<(u8, Vec<u8>)>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut out = Vec::new();
        loop {
            if self.buffer.len() < 5 {
                break;
            }
            let type_byte = self.buffer[0];
            let length = u32::from_be_bytes([
                self.buffer[1],
                self.buffer[2],
                self.buffer[3],
                self.buffer[4],
            ]) as usize;
            if length > self.max_frame_size {
                return Err(format!(
                    "frame length {length} exceeds max frame size {}",
                    self.max_frame_size
                ));
            }
            let total = 5 + length;
            if self.buffer.len() < total {
                break;
            }
            let value = self.buffer[5..total].to_vec();
            self.buffer.drain(..total);
            out.push((type_byte, value));
        }
        Ok(out)
    }
}

fn encode_json_frame(value: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(value).expect("serialize command frame");
    let mut out = Vec::with_capacity(5 + body.len());
    out.push(TYPE_JSON_FRAME);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

fn resolve_env_paths(app: &AppHandle) -> Result<EnvPaths, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dev_gui_root = manifest_dir
        .parent()
        .ok_or("failed to resolve Tauri manifest parent")?
        .to_path_buf();
    let resource_dir = app.path().resource_dir().ok();
    let packaged = !cfg!(debug_assertions);
    let gui_root = if packaged {
        resource_dir.clone().unwrap_or_else(|| dev_gui_root.clone())
    } else {
        dev_gui_root
    };
    let repo_root = if packaged {
        gui_root.clone()
    } else {
        gui_root
            .parent()
            .ok_or("failed to resolve repository root")?
            .to_path_buf()
    };
    let workspace_root = resolve_workspace_root(packaged, resource_dir.as_deref())?;
    let config_path = workspace_root.join(".hawi").join("node_gui.json");
    let backend_log_path = workspace_root.join(".hawi").join("hawi-engine.log");
    let uv_command = resolve_uv_command();
    let engine_launcher =
        resolve_engine_launcher(&repo_root, resource_dir.as_deref(), packaged, &uv_command)?;
    Ok(EnvPaths {
        repo_root,
        workspace_root,
        config_path,
        backend_log_path,
        engine_launcher,
    })
}

fn resolve_workspace_root(
    packaged: bool,
    resources_path: Option<&Path>,
) -> Result<PathBuf, String> {
    if let Some(explicit) = parse_arg_value("--cwd")
        .or_else(|| env::var("HAWI_GUI_CWD").ok())
        .or_else(|| env::var("INIT_CWD").ok())
    {
        return normalize_path(explicit);
    }
    let cwd = env::current_dir()
        .map_err(|error| format!("failed to resolve current directory: {error}"))?;
    if packaged && !is_usable_packaged_workspace_cwd(&cwd, resources_path) {
        return home_dir().ok_or_else(|| "failed to resolve home directory".to_string());
    }
    Ok(cwd)
}

fn is_usable_packaged_workspace_cwd(cwd: &Path, resources_path: Option<&Path>) -> bool {
    if cwd.parent().is_none() {
        return false;
    }
    let Some(resources_path) = resources_path else {
        return true;
    };
    if is_same_path_or_child(cwd, resources_path) {
        return false;
    }
    if let Some(app_root) = find_app_bundle_root(resources_path) {
        if is_same_path_or_child(cwd, &app_root) {
            return false;
        }
    }
    if let Some(install_root) = resources_path.parent() {
        return !is_same_path_or_child(cwd, install_root);
    }
    true
}

fn find_app_bundle_root(candidate: &Path) -> Option<PathBuf> {
    let mut current = candidate.to_path_buf();
    loop {
        if current
            .file_name()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase().ends_with(".app"))
            .unwrap_or(false)
        {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn resolve_uv_command() -> String {
    if let Ok(override_value) = env::var("HAWI_GUI_UV_COMMAND") {
        let trimmed = override_value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    resolve_command_on_path("uv").unwrap_or_else(|| "uv".to_string())
}

fn resolve_command_on_path(command: &str) -> Option<String> {
    let path_value = env::var_os("PATH")?;
    let extensions = if cfg!(windows) {
        vec!["", ".exe", ".cmd", ".bat"]
    } else {
        vec![""]
    };
    for entry in env::split_paths(&path_value) {
        for extension in &extensions {
            let candidate = entry.join(format!("{command}{extension}"));
            if candidate.is_file() {
                return Some(candidate.to_string_lossy().into_owned());
            }
        }
    }
    None
}

fn resolve_engine_launcher(
    repo_root: &Path,
    resources_path: Option<&Path>,
    packaged: bool,
    uv_command: &str,
) -> Result<EngineLauncher, String> {
    if let Ok(override_value) = env::var("HAWI_GUI_ENGINE_COMMAND") {
        let trimmed = override_value.trim();
        if !trimmed.is_empty() {
            return Ok(EngineLauncher {
                command: trimmed.to_string(),
                args_prefix: Vec::new(),
                source: EngineLauncherSource::Bundled,
            });
        }
    }
    if packaged {
        if let Some(command) = resolve_bundled_engine_command(resources_path) {
            return Ok(EngineLauncher {
                command,
                args_prefix: Vec::new(),
                source: EngineLauncherSource::Bundled,
            });
        }
        return Err(
            "Bundled hawi-engine executable was not found in the application resources."
                .to_string(),
        );
    }
    Ok(EngineLauncher {
        command: uv_command.to_string(),
        args_prefix: build_uv_engine_args_prefix(repo_root),
        source: EngineLauncherSource::Uv,
    })
}

fn resolve_bundled_engine_command(resources_path: Option<&Path>) -> Option<String> {
    let resources_path = resources_path?;
    let executable = if cfg!(windows) {
        "hawi-engine.exe"
    } else {
        "hawi-engine"
    };
    let candidates = [
        resources_path.join("bin").join(executable),
        resources_path
            .join("bin")
            .join("hawi-engine")
            .join(executable),
        resources_path
            .join("app.asar.unpacked")
            .join("build")
            .join("bin")
            .join(executable),
        resources_path
            .join("app.asar.unpacked")
            .join("build")
            .join("bin")
            .join("hawi-engine")
            .join(executable),
    ];
    candidates
        .iter()
        .find(|candidate| candidate.is_file())
        .map(|candidate| candidate.to_string_lossy().into_owned())
}

fn build_uv_engine_args_prefix(repo_root: &Path) -> Vec<String> {
    vec![
        "run".to_string(),
        "--project".to_string(),
        repo_root.to_string_lossy().into_owned(),
        "python".to_string(),
        "-m".to_string(),
        "hawi.engine".to_string(),
    ]
}

fn build_engine_run_args(
    repo_root: &Path,
    engine_args: Vec<String>,
    launcher: &EngineLauncher,
) -> Vec<String> {
    let mut args = launcher.args_prefix.clone();
    if launcher.source == EngineLauncherSource::Uv && args.is_empty() {
        args = build_uv_engine_args_prefix(repo_root);
    }
    args.extend(engine_args);
    args
}

fn build_engine_env(repo_root: &Path, launcher: &EngineLauncher) -> HashMap<String, String> {
    let mut vars = env::vars().collect::<HashMap<_, _>>();
    match launcher.source {
        EngineLauncherSource::Bundled => {
            vars.insert("HAWI_GUI_ENGINE_SOURCE".to_string(), "bundled".to_string());
        }
        EngineLauncherSource::Uv => {
            vars.insert("HAWI_GUI_ENGINE_SOURCE".to_string(), "uv".to_string());
            let repo_root = repo_root.to_string_lossy();
            let python_path = vars.get("PYTHONPATH").cloned().unwrap_or_default();
            let next_python_path = if python_path.trim().is_empty() {
                repo_root.into_owned()
            } else {
                format!("{}{}{}", repo_root, path_delimiter(), python_path)
            };
            vars.insert("PYTHONPATH".to_string(), next_python_path);
        }
    }
    vars
}

fn ensure_engine_workspace(env_paths: &EnvPaths) -> Result<(), String> {
    fs::create_dir_all(&env_paths.workspace_root)
        .map_err(|error| format!("failed to create workspace directory: {error}"))?;
    if env_paths
        .workspace_root
        .join(".hawi")
        .join("models.yaml")
        .exists()
    {
        return Ok(());
    }
    let args = build_engine_run_args(
        &env_paths.repo_root,
        vec!["init".to_string()],
        &env_paths.engine_launcher,
    );
    let output = Command::new(&env_paths.engine_launcher.command)
        .args(&args)
        .current_dir(&env_paths.workspace_root)
        .envs(build_engine_env(
            &env_paths.repo_root,
            &env_paths.engine_launcher,
        ))
        .output()
        .map_err(|error| {
            format!(
                "Failed to initialize Hawi workspace with {}: {error}",
                env_paths.engine_launcher.command
            )
        })?;
    if !output.status.success() {
        return Err(format_engine_command_error(
            "Failed to initialize Hawi workspace",
            output.status.code(),
            &output.stderr,
            &output.stdout,
        ));
    }
    Ok(())
}

fn load_inspect_payload(
    repo_root: &Path,
    workspace_root: &Path,
    engine_launcher: &EngineLauncher,
    inspect_args: &[String],
) -> Result<Value, String> {
    let mut engine_args = vec!["--inspect".to_string()];
    engine_args.extend(inspect_args.iter().cloned());
    let args = build_engine_run_args(repo_root, engine_args, engine_launcher);
    let output = Command::new(&engine_launcher.command)
        .args(args)
        .current_dir(workspace_root)
        .envs(build_engine_env(repo_root, engine_launcher))
        .output()
        .map_err(|error| format!("Failed to launch {}: {error}", engine_launcher.command))?;
    if !output.status.success() {
        return Err(format_engine_command_error(
            "Failed to inspect hawi-engine metadata",
            output.status.code(),
            &output.stderr,
            &output.stdout,
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid inspect payload: {error}"))
}

fn format_engine_command_error(
    message: &str,
    status: Option<i32>,
    stderr: &[u8],
    stdout: &[u8],
) -> String {
    let mut details = vec![format!(
        "{message} (exit {}).",
        status.map_or("unknown".to_string(), |value| value.to_string())
    )];
    let stderr = String::from_utf8_lossy(stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(stdout).trim().to_string();
    if !stderr.is_empty() {
        details.push(format!("stderr: {stderr}"));
    }
    if !stdout.is_empty() {
        details.push(format!("stdout: {stdout}"));
    }
    details.join("\n")
}

fn load_config(config_path: &Path, metadata: &Value) -> Value {
    fs::read_to_string(config_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .map(|raw| sanitize_config(&raw, Some(metadata)))
        .unwrap_or_else(|| default_config(metadata))
}

fn default_config(metadata: &Value) -> Value {
    let default_plugins = metadata
        .get("plugin_catalog")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("key").and_then(Value::as_str))
                .filter(|key| *key == "hawi/environ-prompt")
                .map(|key| Value::String(key.to_string()))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({
        "version": 1,
        "modelName": metadata_models(metadata).first().cloned().unwrap_or_default(),
        "systemPrompt": value_string(metadata, "default_system_prompt"),
        "selectedPlugins": default_plugins,
        "pluginConfigs": {},
        "toolCallPurposeEnabled": true,
        "showDebug": true,
    })
}

fn sanitize_config(raw: &Value, metadata: Option<&Value>) -> Value {
    let models = metadata.map(metadata_models).unwrap_or_default();
    let raw_model = value_string(raw, "modelName");
    let model_name = if models.iter().any(|model| model == &raw_model) {
        raw_model
    } else {
        models.first().cloned().unwrap_or_default()
    };

    let plugin_keys = metadata
        .and_then(|value| value.get("plugin_catalog"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("key").and_then(Value::as_str).map(str::to_string))
                .collect::<HashSet<_>>()
        })
        .unwrap_or_default();

    let selected_plugins = string_list(raw.get("selectedPlugins"))
        .into_iter()
        .filter(|key| plugin_keys.contains(key))
        .map(Value::String)
        .collect::<Vec<_>>();
    let plugin_configs = raw
        .get("pluginConfigs")
        .and_then(Value::as_object)
        .map(|configs| {
            configs
                .iter()
                .filter_map(|(key, value)| {
                    if plugin_keys.contains(key) && value.is_object() {
                        Some((key.clone(), value.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Map<_, _>>()
        })
        .unwrap_or_default();
    let system_prompt = non_empty_string(raw.get("systemPrompt"))
        .or_else(|| metadata.map(|value| value_string(value, "default_system_prompt")))
        .unwrap_or_default();

    json!({
        "version": 1,
        "modelName": model_name,
        "systemPrompt": system_prompt,
        "selectedPlugins": selected_plugins,
        "pluginConfigs": plugin_configs,
        "toolCallPurposeEnabled": raw.get("toolCallPurposeEnabled").and_then(Value::as_bool).unwrap_or(true),
        "showDebug": raw.get("showDebug").and_then(Value::as_bool).unwrap_or(false),
    })
}

fn save_config_file(config_path: &Path, next_config: &Value) -> Result<(), String> {
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create config directory: {error}"))?;
    }
    fs::write(
        config_path,
        serde_json::to_vec_pretty(next_config).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("failed to save config: {error}"))
}

fn preserve_provider_order(previous_models: Vec<String>, next_models: Vec<String>) -> Vec<String> {
    let previous_providers = provider_order(&previous_models);
    let mut grouped_next = group_models_by_provider(next_models);
    let mut ordered = Vec::new();
    for provider in previous_providers {
        if let Some(models) = grouped_next.remove(&provider) {
            ordered.extend(models);
        }
    }
    for (_, models) in grouped_next {
        ordered.extend(models);
    }
    ordered
}

fn provider_order(models: &[String]) -> Vec<String> {
    let mut providers = Vec::new();
    let mut seen = HashSet::new();
    for model in models {
        let provider = model_provider(model);
        if !provider.is_empty() && seen.insert(provider.clone()) {
            providers.push(provider);
        }
    }
    providers
}

fn group_models_by_provider(models: Vec<String>) -> HashMap<String, Vec<String>> {
    let mut groups: HashMap<String, Vec<String>> = HashMap::new();
    let mut seen = HashSet::new();
    for model in models {
        if !seen.insert(model.clone()) {
            continue;
        }
        let provider = model_provider(&model);
        if provider.is_empty() {
            continue;
        }
        groups.entry(provider).or_default().push(model);
    }
    groups
}

fn model_provider(model: &str) -> String {
    model.split('/').next().unwrap_or_default().to_string()
}

fn tool_call_purpose_engine_args(enabled: bool) -> Vec<String> {
    if !enabled {
        return Vec::new();
    }
    let directive = json!({
        "name": "tool_call_purpose",
        "schema": {
            "type": "string",
            "default": Value::Null,
            "description": "【必填】用一句话说明本次工具调用的目的；允许与其他调用重复，会显示在工具标题旁边。未指定时工具仍会执行，但结果会附加错误提示，说明这会导致用户误解并影响自动审核 agent 的判断准确度。"
        },
        "required": true
    });
    vec![
        "--extra-tool-parameter-json".to_string(),
        directive.to_string(),
    ]
}

impl Drop for CoreProcess {
    fn drop(&mut self) {
        self.stop("drop");
    }
}

struct EngineRecord {
    session_id: String,
    core: CoreProcess,
    launch_profile: Value,
    workspace_root: PathBuf,
    loaded_at: u64,
    last_finished_at: Option<u64>,
    has_visible_messages: bool,
    agent_state: String,
    runner_state: String,
    suppress_events: bool,
    stopping: bool,
}

struct SessionEngineManager {
    event_tx: mpsc::Sender<ManagerEngineEvent>,
    app: AppHandle,
    loaded: HashMap<String, EngineRecord>,
    current_session_id: Option<String>,
    metadata: Option<Value>,
    default_config: Option<Value>,
    refreshed_providers: HashSet<String>,
    enforcing_limit: bool,
    readonly_core: Option<CoreProcess>,
    repo_root: PathBuf,
    workspace_root: PathBuf,
    backend_log_path: PathBuf,
    engine_launcher: EngineLauncher,
}

#[derive(Default)]
struct StartRecordOptions {
    initial_session_id: Option<String>,
    initial_session_name: Option<String>,
    suppress_events: bool,
    workspace_root: Option<PathBuf>,
}

impl SessionEngineManager {
    fn new_shared(
        app: AppHandle,
        repo_root: PathBuf,
        workspace_root: PathBuf,
        backend_log_path: PathBuf,
        engine_launcher: EngineLauncher,
    ) -> Arc<Mutex<Self>> {
        let (event_tx, event_rx) = mpsc::channel::<ManagerEngineEvent>();
        let manager = Arc::new(Mutex::new(Self {
            event_tx,
            app,
            loaded: HashMap::new(),
            current_session_id: None,
            metadata: None,
            default_config: None,
            refreshed_providers: HashSet::new(),
            enforcing_limit: false,
            readonly_core: None,
            repo_root,
            workspace_root,
            backend_log_path,
            engine_launcher,
        }));
        let weak = Arc::downgrade(&manager);
        thread::spawn(move || {
            while let Ok(event) = event_rx.recv() {
                let Some(manager) = weak.upgrade() else {
                    break;
                };
                if let Ok(mut manager) = manager.lock() {
                    manager.handle_engine_emit(&event.session_id, event.channel, event.payload);
                };
            }
        });
        manager
    }

    fn configure(&mut self, metadata: Value, config: Value, refreshed_providers: &HashSet<String>) {
        self.metadata = Some(metadata);
        self.default_config = Some(config);
        self.refreshed_providers = refreshed_providers.clone();
    }

    fn snapshot(&self) -> Value {
        json!({
            "currentSessionId": self.current_session_id,
            "currentWorkspaceRoot": self.current_workspace_root(),
            "runningSessionCount": self.running_session_count(),
            "loadedSessionCount": self.visible_loaded_session_count(),
            "maxLoadedSessions": MAX_LOADED_SESSIONS,
            "coreRunning": !self.loaded.is_empty(),
        })
    }

    fn stop_all(&mut self, reason: &str) {
        let session_ids = self.loaded.keys().cloned().collect::<Vec<_>>();
        for session_id in session_ids {
            self.stop_record_by_id(&session_id, reason);
        }
        if let Some(mut core) = self.readonly_core.take() {
            core.stop(reason);
        }
    }

    fn restart_current(&mut self, config: Value) -> Result<(), String> {
        self.default_config = Some(config.clone());
        let session_id = self
            .current_session_id
            .clone()
            .unwrap_or_else(generate_session_id);
        if self.loaded.contains_key(&session_id) {
            self.stop_record_by_id(&session_id, "restart");
        }
        let profile = profile_from_config(&config);
        self.current_session_id = Some(session_id.clone());
        self.start_record(&session_id, profile, StartRecordOptions::default())?;
        let load_result = {
            let record = self
                .loaded
                .get_mut(&session_id)
                .expect("record just started");
            record.core.send_command(
                "session_load",
                json!({ "session_id": session_id }),
                SESSION_COMMAND_TIMEOUT_MS,
            )
        };
        match load_result {
            Ok(_) => {
                if let Some(record) = self.loaded.get_mut(&session_id) {
                    record.has_visible_messages = true;
                }
                self.emit_session_runtime_status(&session_id, None);
                self.enforce_loaded_limit();
                Ok(())
            }
            Err(error) if is_missing_session_error(&error) => {
                self.stop_record_by_id(&session_id, "restart-load-failed");
                self.start_record(
                    &session_id,
                    profile_from_config(&config),
                    StartRecordOptions {
                        initial_session_id: Some(session_id.clone()),
                        initial_session_name: Some(session_id.clone()),
                        ..StartRecordOptions::default()
                    },
                )?;
                self.emit_session_runtime_status(&session_id, None);
                Ok(())
            }
            Err(error) => {
                self.stop_record_by_id(&session_id, "restart-load-failed");
                Err(error.message)
            }
        }
    }

    fn send_command(
        &mut self,
        command_type: &str,
        payload: Value,
        target_session_id: Option<String>,
    ) -> Result<Value, String> {
        let payload = Value::Object(object_or_empty(payload));
        match command_type {
            "session_list" => self.session_list_frame(),
            "session_new" => self.create_session(payload),
            "session_fork" => self.fork_session(payload),
            "session_switch" | "session_load" => self.switch_session(payload),
            "session_delete" => self.delete_session(payload),
            "session_rename" => self.rename_session(payload),
            "change_cwd" => self.change_workspace(payload),
            _ if is_readonly_command(command_type, &payload) => {
                self.readonly_command(command_type, payload)
            }
            _ => self.route_command(command_type, payload, target_session_id),
        }
    }

    fn refresh_models(&mut self, provider: &str) -> Result<Option<Value>, String> {
        let session_id = self
            .current_session_id
            .clone()
            .or_else(|| self.loaded.keys().next().cloned());
        let Some(session_id) = session_id else {
            return Ok(None);
        };
        let record = self
            .loaded
            .get_mut(&session_id)
            .ok_or_else(|| format!("Session is not loaded: {session_id}"))?;
        record
            .core
            .send_command("refresh_models", json!({ "provider": provider }), 60_000)
            .map(Some)
            .map_err(|error| error.message)
    }

    fn route_command(
        &mut self,
        command_type: &str,
        payload: Value,
        target_session_id: Option<String>,
    ) -> Result<Value, String> {
        let session_id = target_session_id
            .or_else(|| {
                payload
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
            })
            .or_else(|| self.current_session_id.clone())
            .ok_or_else(|| "No active session".to_string())?;
        if !self.loaded.contains_key(&session_id) {
            if command_type == "session_history" || command_type == "session_export_markdown" {
                let catalog_id = self.catalog_record()?;
                let target = payload
                    .get("session_id")
                    .and_then(Value::as_str)
                    .unwrap_or(&session_id)
                    .to_string();
                let mut next_payload = object_or_empty(payload);
                next_payload.insert("session_id".to_string(), Value::String(target));
                let record = self.loaded.get_mut(&catalog_id).expect("catalog record");
                return record
                    .core
                    .send_command(
                        command_type,
                        Value::Object(next_payload),
                        command_timeout(command_type),
                    )
                    .map_err(|error| error.message);
            }
            return Err(format!("Session is not loaded: {session_id}"));
        }

        let frame = {
            let record = self.loaded.get_mut(&session_id).expect("record exists");
            record
                .core
                .send_command(command_type, payload.clone(), command_timeout(command_type))
        }
        .map_err(|error| error.message)?;

        if command_type == "set_system_prompt" {
            if let Some(system_prompt) = payload.get("system_prompt").and_then(Value::as_str) {
                let profile = {
                    let record = self.loaded.get_mut(&session_id).expect("record exists");
                    set_value_field(
                        &mut record.launch_profile,
                        "systemPrompt",
                        Value::String(system_prompt.to_string()),
                    );
                    record.launch_profile.clone()
                };
                self.sync_default_config_from_profile(&profile)?;
                self.save_session_profile(&session_id);
            }
        } else if command_type == "switch_model" {
            if let Some(model_name) = payload.get("model_name").and_then(Value::as_str) {
                let profile = {
                    let record = self.loaded.get_mut(&session_id).expect("record exists");
                    set_value_field(
                        &mut record.launch_profile,
                        "modelName",
                        Value::String(model_name.to_string()),
                    );
                    record.launch_profile.clone()
                };
                self.sync_default_config_from_profile(&profile)?;
                self.save_session_profile(&session_id);
            }
        } else if command_type == "apply_plugins" {
            let selected_plugins = Value::Array(
                string_list(payload.get("selected_plugins"))
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            );
            let plugin_configs = payload
                .get("plugin_configs")
                .filter(|value| value.is_object())
                .cloned()
                .unwrap_or_else(|| json!({}));
            let profile = {
                let record = self.loaded.get_mut(&session_id).expect("record exists");
                set_value_field(
                    &mut record.launch_profile,
                    "selectedPlugins",
                    selected_plugins,
                );
                set_value_field(&mut record.launch_profile, "pluginConfigs", plugin_configs);
                record.launch_profile.clone()
            };
            self.sync_default_config_from_profile(&profile)?;
            self.save_session_profile(&session_id);
        }

        self.emit_session_runtime_status(&session_id, None);
        Ok(frame)
    }

    fn create_session(&mut self, payload: Value) -> Result<Value, String> {
        let workspace_root = self.current_workspace_root();
        self.save_current_session();
        self.discard_current_empty_session();
        let session_id = generate_session_id();
        let name = payload
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&session_id)
            .to_string();
        let profile = launch_profile_from_unknown(payload.get("gui_launch_profile"))
            .or_else(|| launch_profile_from_unknown(payload.get("launch_profile")))
            .unwrap_or_else(|| {
                profile_from_config(self.require_default_config().expect("config ready"))
            });
        self.sync_default_config_from_profile(&profile)?;
        self.current_session_id = Some(session_id.clone());
        self.start_record(
            &session_id,
            profile,
            StartRecordOptions {
                initial_session_id: Some(session_id.clone()),
                initial_session_name: Some(name.clone()),
                workspace_root: Some(workspace_root),
                ..StartRecordOptions::default()
            },
        )?;
        self.emit_session_runtime_status(&session_id, None);
        self.enforce_loaded_limit();
        Ok(ack_frame(
            "session_new",
            json!({ "session_id": session_id, "name": name }),
        ))
    }

    fn change_workspace(&mut self, payload: Value) -> Result<Value, String> {
        let target = payload
            .get("cwd")
            .or_else(|| payload.get("path"))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "'cwd' is required".to_string())
            .and_then(|value| normalize_path(PathBuf::from(value.trim())))?;
        let previous_workspace_root = self.current_workspace_root();
        if same_workspace_root(&previous_workspace_root, &target) {
            return Ok(ack_frame(
                "change_cwd",
                json!({
                    "session_id": self.current_session_id.clone(),
                    "workspace_switched": false,
                    "previous_cwd": previous_workspace_root,
                    "last_cwd": target,
                }),
            ));
        }

        self.save_current_session();
        self.discard_current_empty_session();
        if let Some(mut core) = self.readonly_core.take() {
            core.stop("change-cwd");
        }

        self.workspace_root = target.clone();
        let session_id = generate_session_id();
        let profile = profile_from_config(self.require_default_config()?);
        self.current_session_id = Some(session_id.clone());
        self.start_record(
            &session_id,
            profile,
            StartRecordOptions {
                initial_session_id: Some(session_id.clone()),
                initial_session_name: Some(session_id.clone()),
                workspace_root: Some(target.clone()),
                ..StartRecordOptions::default()
            },
        )?;
        self.emit_session_runtime_status(&session_id, None);
        self.emit_workspace_changed_with_message(
            &session_id,
            &previous_workspace_root,
            &target,
            format!(
                "已切换工作目录：{} -> {}",
                previous_workspace_root.display(),
                target.display()
            ),
        );
        self.enforce_loaded_limit();
        Ok(ack_frame(
            "change_cwd",
            json!({
                "session_id": session_id,
                "workspace_switched": true,
                "previous_cwd": previous_workspace_root,
                "last_cwd": target,
            }),
        ))
    }

    fn fork_session(&mut self, payload: Value) -> Result<Value, String> {
        let source_session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| self.current_session_id.clone())
            .ok_or_else(|| "No session available to fork".to_string())?;
        self.save_current_session();
        let source_meta = self.find_session_meta(&source_session_id)?;
        let source_profile = source_meta
            .as_ref()
            .and_then(|meta| launch_profile_from_unknown(meta.get("gui_launch_profile")))
            .or_else(|| {
                self.loaded
                    .get(&source_session_id)
                    .map(|record| record.launch_profile.clone())
            })
            .unwrap_or_else(|| {
                profile_from_config(self.require_default_config().expect("config ready"))
            });
        let source_workspace_root = source_meta
            .as_ref()
            .and_then(workspace_root_from_meta)
            .or_else(|| {
                self.loaded
                    .get(&source_session_id)
                    .map(|record| record.workspace_root.clone())
            })
            .unwrap_or_else(|| self.current_workspace_root());
        let provisional_id = format!("forking-{}-{}", base36(now_ms()), random_hex_suffix(6));
        self.start_record(
            &provisional_id,
            source_profile,
            StartRecordOptions {
                suppress_events: true,
                workspace_root: Some(source_workspace_root),
                ..StartRecordOptions::default()
            },
        )?;
        let frame = {
            let record = self
                .loaded
                .get_mut(&provisional_id)
                .expect("provisional record");
            record
                .core
                .send_command("session_fork", payload, FORK_COMMAND_TIMEOUT_MS)
        };
        let frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                self.stop_record_by_id(&provisional_id, "fork-failed");
                return Err(error.message);
            }
        };
        let forked_session_id = frame_payload(&frame)
            .get("session_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "Fork did not return a session id".to_string())?
            .to_string();
        let mut record = self
            .loaded
            .remove(&provisional_id)
            .ok_or_else(|| "Fork engine disappeared".to_string())?;
        record.session_id = forked_session_id.clone();
        record.suppress_events = false;
        record.loaded_at = now_ms();
        record.has_visible_messages = true;
        self.loaded.insert(forked_session_id.clone(), record);
        self.current_session_id = Some(forked_session_id.clone());
        self.emit_session_runtime_status(&forked_session_id, None);
        self.enforce_loaded_limit();
        Ok(frame)
    }

    fn switch_session(&mut self, payload: Value) -> Result<Value, String> {
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "'session_id' is required".to_string())?
            .to_string();
        if self.loaded.contains_key(&session_id) {
            let previous_workspace_root = self.current_workspace_root();
            self.save_current_session();
            if self.current_session_id.as_deref() != Some(&session_id) {
                self.discard_current_empty_session();
            }
            self.current_session_id = Some(session_id.clone());
            self.emit_session_runtime_status(&session_id, None);
            let next_workspace_root = self
                .loaded
                .get(&session_id)
                .map(|record| record.workspace_root.clone())
                .unwrap_or_else(|| previous_workspace_root.clone());
            let workspace_switched =
                !same_workspace_root(&previous_workspace_root, &next_workspace_root);
            if workspace_switched {
                self.emit_workspace_changed(
                    &session_id,
                    &previous_workspace_root,
                    &next_workspace_root,
                );
            }
            self.enforce_loaded_limit();
            return Ok(ack_frame(
                "session_switch",
                json!({
                    "session_id": session_id,
                    "already_loaded": true,
                    "workspace_switched": workspace_switched,
                    "previous_cwd": previous_workspace_root,
                    "last_cwd": next_workspace_root,
                }),
            ));
        }

        let meta = self.find_session_meta(&session_id)?;
        let previous_workspace_root = self.current_workspace_root();
        self.save_current_session();
        self.discard_current_empty_session();
        let target_workspace_root = meta
            .as_ref()
            .and_then(workspace_root_from_meta)
            .unwrap_or_else(|| previous_workspace_root.clone());
        let workspace_switched =
            !same_workspace_root(&previous_workspace_root, &target_workspace_root);
        let profile = meta
            .as_ref()
            .and_then(|meta| launch_profile_from_unknown(meta.get("gui_launch_profile")))
            .unwrap_or_else(|| {
                profile_from_config(self.require_default_config().expect("config ready"))
            });
        self.start_record(
            &session_id,
            profile,
            StartRecordOptions {
                workspace_root: Some(target_workspace_root.clone()),
                ..StartRecordOptions::default()
            },
        )?;
        let frame = {
            let record = self.loaded.get_mut(&session_id).expect("record started");
            record.core.send_command(
                "session_load",
                json!({ "session_id": session_id }),
                SESSION_COMMAND_TIMEOUT_MS,
            )
        };
        let mut frame = match frame {
            Ok(frame) => frame,
            Err(error) => {
                self.stop_record_by_id(&session_id, "load-failed");
                return Err(error.message);
            }
        };
        if let Some(record) = self.loaded.get_mut(&session_id) {
            record.has_visible_messages = true;
        }
        self.current_session_id = Some(session_id.clone());
        self.emit_session_runtime_status(&session_id, None);
        if workspace_switched {
            self.emit_workspace_changed(
                &session_id,
                &previous_workspace_root,
                &target_workspace_root,
            );
        }
        self.enforce_loaded_limit();
        merge_payload(
            &mut frame,
            json!({
                "command": "session_switch",
                "session_id": session_id,
                "workspace_switched": workspace_switched,
                "previous_cwd": previous_workspace_root,
                "last_cwd": target_workspace_root,
            }),
        );
        Ok(frame)
    }

    fn delete_session(&mut self, payload: Value) -> Result<Value, String> {
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "'session_id' is required".to_string())?
            .to_string();
        if self
            .loaded
            .get(&session_id)
            .is_some_and(|record| is_running_agent_state(&record.agent_state))
        {
            return Err("Cannot delete a running session.".to_string());
        }
        let was_current = self.current_session_id.as_deref() == Some(&session_id);
        let next_current = if was_current {
            self.next_current_session_id_after_delete(&session_id)
        } else {
            self.current_session_id.clone()
        };
        if self.loaded.contains_key(&session_id) {
            self.stop_record_by_id(&session_id, "delete-session");
        }
        if was_current {
            self.current_session_id = next_current.clone();
            if let Some(next_current) = &next_current {
                self.emit_session_runtime_status(next_current, None);
            }
        }
        let catalog_id = self.catalog_record()?;
        let frame = {
            let catalog = self.loaded.get_mut(&catalog_id).expect("catalog record");
            catalog.core.send_command(
                "session_delete",
                json!({ "session_id": session_id }),
                SESSION_COMMAND_TIMEOUT_MS,
            )
        }
        .map_err(|error| error.message)?;
        Ok(frame_with_payload(
            frame,
            json!({
                "session_id": session_id,
                "current_session_id": self.current_session_id,
                "running_session_count": self.running_session_count(),
                "loaded_session_count": self.visible_loaded_session_count(),
                "max_loaded_sessions": MAX_LOADED_SESSIONS,
            }),
        ))
    }

    fn rename_session(&mut self, payload: Value) -> Result<Value, String> {
        let session_id = payload
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "'session_id' is required".to_string())?
            .to_string();
        let name = payload
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "'name' is required".to_string())?
            .to_string();
        let target_id = if self.loaded.contains_key(&session_id) {
            session_id.clone()
        } else {
            self.catalog_record()?
        };
        let frame = {
            let target = self.loaded.get_mut(&target_id).expect("target record");
            target.core.send_command(
                "session_rename",
                json!({ "session_id": session_id, "name": name }),
                SESSION_COMMAND_TIMEOUT_MS,
            )
        }
        .map_err(|error| error.message)?;
        self.emit_session_runtime_status(&session_id, None);
        Ok(frame_with_payload(
            frame,
            json!({
                "session_id": session_id,
                "name": name,
                "current_session_id": self.current_session_id,
                "running_session_count": self.running_session_count(),
                "loaded_session_count": self.visible_loaded_session_count(),
                "max_loaded_sessions": MAX_LOADED_SESSIONS,
            }),
        ))
    }

    fn start_record(
        &mut self,
        session_id: &str,
        launch_profile: Value,
        options: StartRecordOptions,
    ) -> Result<(), String> {
        if self.loaded.contains_key(session_id) {
            return Ok(());
        }
        let metadata = self.require_metadata()?.clone();
        let config =
            config_from_profile(&launch_profile, self.require_default_config()?, &metadata);
        let workspace_root = options
            .workspace_root
            .unwrap_or_else(|| self.current_workspace_root());
        let emit_session_id = session_id.to_string();
        let event_tx = self.event_tx.clone();
        let emit: CoreEmit = Arc::new(move |channel, payload| {
            let _ = event_tx.send(ManagerEngineEvent {
                session_id: emit_session_id.clone(),
                channel,
                payload,
            });
        });
        let core = CoreProcess::new(
            emit,
            self.repo_root.clone(),
            workspace_root.clone(),
            self.log_path_for_session(session_id, &workspace_root),
            self.engine_launcher.clone(),
        );
        self.loaded.insert(
            session_id.to_string(),
            EngineRecord {
                session_id: session_id.to_string(),
                core,
                launch_profile: launch_profile.clone(),
                workspace_root,
                loaded_at: now_ms(),
                last_finished_at: None,
                has_visible_messages: false,
                agent_state: "IDLE".to_string(),
                runner_state: "IDLE".to_string(),
                suppress_events: options.suppress_events,
                stopping: false,
            },
        );
        let spawn_payload = {
            let record = self.loaded.get_mut(session_id).expect("record inserted");
            record.core.start(
                &config,
                &metadata,
                &self.refreshed_providers,
                CoreStartOptions {
                    initial_session_id: options.initial_session_id,
                    initial_session_name: options.initial_session_name,
                    launch_profile: Some(launch_profile),
                },
            )
        };
        match spawn_payload {
            Ok(payload) => {
                self.handle_engine_emit(session_id, "core:spawn".to_string(), payload);
                Ok(())
            }
            Err(error) => {
                self.loaded.remove(session_id);
                Err(error)
            }
        }
    }

    fn stop_record_by_id(&mut self, session_id: &str, reason: &str) {
        let Some(record) = self.loaded.get_mut(session_id) else {
            return;
        };
        if record.stopping {
            return;
        }
        record.stopping = true;
        if record.core.is_running() {
            let _ =
                record
                    .core
                    .send_command("session_save_now", json!({}), SESSION_COMMAND_TIMEOUT_MS);
        }
        record.core.stop(reason);
        self.loaded.remove(session_id);
        self.emit_session_runtime_status(session_id, Some("unloaded"));
    }

    fn save_session_profile(&mut self, session_id: &str) {
        let Some(record) = self.loaded.get_mut(session_id) else {
            return;
        };
        if record.core.is_running() {
            let _ =
                record
                    .core
                    .send_command("session_save_now", json!({}), SESSION_COMMAND_TIMEOUT_MS);
        }
    }

    fn save_current_session(&mut self) {
        let Some(current) = self.current_session_id.clone() else {
            return;
        };
        self.save_session_profile(&current);
    }

    fn discard_current_empty_session(&mut self) {
        let Some(current) = self.current_session_id.clone() else {
            return;
        };
        let Some(record) = self.loaded.get(&current) else {
            return;
        };
        if record.has_visible_messages || is_running_agent_state(&record.agent_state) {
            return;
        }
        self.current_session_id = None;
        self.stop_record_by_id(&current, "replace-empty-session");
    }

    fn handle_engine_emit(&mut self, session_id: &str, channel: String, payload: Value) {
        if channel == "core:event" && is_core_frame(&payload) {
            let suppress = self
                .loaded
                .get(session_id)
                .is_some_and(|record| record.suppress_events);
            if suppress {
                return;
            }
            let frame = inject_session_id(payload, session_id);
            if let Some(record) = self.loaded.get_mut(session_id) {
                update_record_from_frame(record, &frame);
            }
            let _ = self.app.emit("core:event", frame.clone());
            self.emit_session_runtime_status(session_id, None);
            if frame_type(&frame) == Some("run.stop") || frame_type(&frame) == Some("core.status") {
                self.enforce_loaded_limit();
            }
            return;
        }
        if channel == "core:stderr" {
            let _ = self.app.emit(
                "core:stderr",
                format!(
                    "[{}] {}",
                    short_session_id(session_id),
                    payload.as_str().unwrap_or("")
                ),
            );
            return;
        }
        if channel == "core:spawn" {
            let mut next = object_or_empty(payload);
            next.insert(
                "session_id".to_string(),
                Value::String(session_id.to_string()),
            );
            let _ = self.app.emit("core:spawn", next);
            return;
        }
        if channel == "core:exit" {
            if self.loaded.remove(session_id).is_some() {
                let mut next = object_or_empty(payload);
                next.insert(
                    "session_id".to_string(),
                    Value::String(session_id.to_string()),
                );
                let _ = self.app.emit("core:exit", next);
                self.emit_session_runtime_status(session_id, Some("unloaded"));
            }
            return;
        }
        let _ = self.app.emit(&channel, payload);
    }

    fn session_list_frame(&mut self) -> Result<Value, String> {
        let base_sessions = self.read_session_catalog().unwrap_or_default();
        let mut by_id: HashMap<String, Value> = HashMap::new();
        for session in base_sessions {
            let Some(session_id) = session
                .get("session_id")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
            else {
                continue;
            };
            let mut normalized = normalize_session_meta(&session);
            set_value_field(
                &mut normalized,
                "load_state",
                Value::String("unloaded".to_string()),
            );
            let launch_profile = launch_profile_from_unknown(normalized.get("gui_launch_profile"))
                .unwrap_or(Value::Null);
            set_value_field(&mut normalized, "gui_launch_profile", launch_profile);
            by_id.insert(session_id, normalized);
        }
        for record in self.loaded.values_mut() {
            let existing = by_id.get(&record.session_id).cloned();
            if existing.is_none() && !record.has_visible_messages {
                continue;
            }
            if existing.is_some() {
                record.has_visible_messages = true;
            }
            let mut next = existing.unwrap_or_else(|| json!({}));
            let name = next
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or(&record.session_id)
                .to_string();
            let created_at = next
                .get("created_at")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| iso_timestamp(record.loaded_at));
            let updated_at = next
                .get("updated_at")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| iso_timestamp(record.loaded_at));
            let last_checkpoint_event = next
                .get("last_checkpoint_event")
                .cloned()
                .unwrap_or(Value::Null);
            let components_present = next
                .get("components_present")
                .cloned()
                .unwrap_or_else(|| json!([]));
            merge_payload(
                &mut next,
                json!({
                    "session_id": record.session_id,
                    "name": name,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "last_checkpoint_event": last_checkpoint_event,
                    "components_present": components_present,
                    "locked": false,
                    "lock_owner": Value::Null,
                    "load_state": load_state_for_record(record),
                    "loaded_at": record.loaded_at,
                    "last_finished_at": record.last_finished_at,
                    "gui_launch_profile": record.launch_profile,
                    "last_cwd": record.workspace_root,
                }),
            );
            by_id.insert(record.session_id.clone(), next);
        }
        let mut sessions = by_id.into_values().collect::<Vec<_>>();
        sessions.sort_by(compare_sessions_by_created_at);
        Ok(ack_frame(
            "session_list",
            json!({
                "sessions": sessions,
                "current_session_id": self.current_session_id,
                "running_session_count": self.running_session_count(),
                "loaded_session_count": self.visible_loaded_session_count(),
                "max_loaded_sessions": MAX_LOADED_SESSIONS,
            }),
        ))
    }

    fn read_session_catalog(&mut self) -> Result<Vec<Value>, String> {
        let frame = self.readonly_command("session_list", json!({}))?;
        Ok(frame_payload(&frame)
            .get("sessions")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|value| normalize_session_meta(&value))
            .collect())
    }

    fn find_session_meta(&mut self, session_id: &str) -> Result<Option<Value>, String> {
        Ok(self
            .read_session_catalog()
            .unwrap_or_default()
            .into_iter()
            .find(|session| session.get("session_id").and_then(Value::as_str) == Some(session_id)))
    }

    fn catalog_record(&mut self) -> Result<String, String> {
        if let Some(current) = self.current_session_id.clone() {
            if self.loaded.contains_key(&current) {
                return Ok(current);
            }
        }
        if let Some(first) = self.loaded.keys().next().cloned() {
            return Ok(first);
        }
        let session_id = format!("catalog-{}", generate_session_id());
        let profile = profile_from_config(self.require_default_config()?);
        self.start_record(
            &session_id,
            profile,
            StartRecordOptions {
                suppress_events: true,
                workspace_root: Some(self.current_workspace_root()),
                ..StartRecordOptions::default()
            },
        )?;
        Ok(session_id)
    }

    fn readonly_command(&mut self, command_type: &str, payload: Value) -> Result<Value, String> {
        let mut next_payload = object_or_empty(payload);
        next_payload.remove("read_only");
        let core = self.ensure_readonly_core()?;
        core.send_command(
            command_type,
            Value::Object(next_payload),
            command_timeout(command_type),
        )
        .map_err(|error| error.message)
    }

    fn ensure_readonly_core(&mut self) -> Result<&mut CoreProcess, String> {
        if self
            .readonly_core
            .as_ref()
            .is_some_and(|core| core.is_running())
        {
            return self
                .readonly_core
                .as_mut()
                .ok_or_else(|| "readonly engine disappeared".to_string());
        }

        let app = self.app.clone();
        let emit: CoreEmit = Arc::new(move |channel, payload| {
            emit_readonly_engine_event(&app, channel, payload);
        });
        let mut core = CoreProcess::new(
            emit,
            self.repo_root.clone(),
            self.current_workspace_root(),
            self.readonly_log_path(),
            self.engine_launcher.clone(),
        );
        let spawn_payload = core.start_readonly()?;
        emit_readonly_engine_event(&self.app, "core:spawn".to_string(), spawn_payload);
        self.readonly_core = Some(core);
        self.readonly_core
            .as_mut()
            .ok_or_else(|| "readonly engine disappeared".to_string())
    }

    fn enforce_loaded_limit(&mut self) {
        if self.enforcing_limit {
            return;
        }
        self.enforcing_limit = true;
        while self.loaded.len() > MAX_LOADED_SESSIONS {
            let candidate = self
                .loaded
                .values()
                .filter(|record| {
                    self.current_session_id.as_deref() != Some(&record.session_id)
                        && !is_running_agent_state(&record.agent_state)
                        && record.core.is_running()
                })
                .min_by_key(|record| record.last_finished_at.unwrap_or(record.loaded_at))
                .map(|record| record.session_id.clone());
            let Some(candidate) = candidate else {
                break;
            };
            self.stop_record_by_id(&candidate, "max-loaded-sessions");
        }
        self.enforcing_limit = false;
    }

    fn emit_session_runtime_status(&self, session_id: &str, override_state: Option<&str>) {
        let record = self.loaded.get(session_id);
        let _ = self.app.emit(
            "core:event",
            json!({
                "version": VERSION,
                "type": "gui.session_status",
                "payload": {
                    "session_id": session_id,
                    "load_state": override_state.unwrap_or_else(|| record.map(load_state_for_record).unwrap_or("unloaded")),
                    "loaded_at": record.map(|record| record.loaded_at),
                    "last_finished_at": record.and_then(|record| record.last_finished_at),
                    "has_visible_messages": record.map(|record| record.has_visible_messages).unwrap_or(false),
                    "last_cwd": record.map(|record| record.workspace_root.clone()),
                    "current_session_id": self.current_session_id,
                    "running_session_count": self.running_session_count(),
                    "loaded_session_count": self.visible_loaded_session_count(),
                    "max_loaded_sessions": MAX_LOADED_SESSIONS,
                },
            }),
        );
    }

    fn running_session_count(&self) -> usize {
        self.loaded
            .values()
            .filter(|record| is_running_agent_state(&record.agent_state))
            .count()
    }

    fn visible_loaded_session_count(&self) -> usize {
        self.loaded
            .values()
            .filter(|record| record.has_visible_messages)
            .count()
    }

    fn current_workspace_root(&self) -> PathBuf {
        self.current_session_id
            .as_ref()
            .and_then(|id| self.loaded.get(id))
            .map(|record| record.workspace_root.clone())
            .unwrap_or_else(|| self.workspace_root.clone())
    }

    fn emit_workspace_changed(&self, session_id: &str, previous: &Path, next: &Path) {
        self.emit_workspace_changed_with_message(
            session_id,
            previous,
            next,
            format!(
                "已根据 Session 记录切换工作目录：{} -> {}",
                previous.display(),
                next.display()
            ),
        );
    }

    fn emit_workspace_changed_with_message(
        &self,
        session_id: &str,
        previous: &Path,
        next: &Path,
        message: String,
    ) {
        let _ = self.app.emit(
            "core:event",
            json!({
                "version": VERSION,
                "type": "gui.workspace_changed",
                "payload": {
                    "session_id": session_id,
                    "previous_cwd": previous,
                    "last_cwd": next,
                    "message": message,
                },
            }),
        );
    }

    fn log_path_for_session(&self, session_id: &str, workspace_root: &Path) -> PathBuf {
        let stem = self
            .backend_log_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("hawi-engine");
        let ext = self
            .backend_log_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("log");
        workspace_root
            .join(".hawi")
            .join(format!("{}-{}.{}", stem, safe_filename(session_id), ext))
    }

    fn readonly_log_path(&self) -> PathBuf {
        let stem = self
            .backend_log_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("hawi-engine");
        let ext = self
            .backend_log_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("log");
        self.current_workspace_root()
            .join(".hawi")
            .join(format!("{stem}-readonly.{ext}"))
    }

    fn require_metadata(&self) -> Result<&Value, String> {
        self.metadata
            .as_ref()
            .ok_or_else(|| "GUI metadata is not ready".to_string())
    }

    fn require_default_config(&self) -> Result<&Value, String> {
        self.default_config
            .as_ref()
            .ok_or_else(|| "GUI config is not ready".to_string())
    }

    fn sync_default_config_from_profile(&mut self, profile: &Value) -> Result<(), String> {
        let config = config_from_profile(
            profile,
            self.require_default_config()?,
            self.require_metadata()?,
        );
        self.default_config = Some(config);
        Ok(())
    }

    fn next_current_session_id_after_delete(&self, deleted_session_id: &str) -> Option<String> {
        self.loaded
            .values()
            .filter(|record| record.session_id != deleted_session_id && record.has_visible_messages)
            .max_by_key(|record| record.loaded_at)
            .map(|record| record.session_id.clone())
    }
}

impl Drop for SessionEngineManager {
    fn drop(&mut self) {
        self.stop_all("manager-drop");
    }
}

#[tauri::command]
fn get_metadata(state: State<'_, GuiState>) -> Result<Value, String> {
    let inspect = state
        .inspect
        .lock()
        .map_err(|_| "inspect lock poisoned".to_string())?
        .clone();
    let config = state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())?
        .clone();
    let snapshot = state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?
        .snapshot();
    Ok(gui_metadata(inspect, config, snapshot))
}

#[tauri::command]
fn save_config(config: Value, state: State<'_, GuiState>) -> Result<Value, String> {
    let inspect = state
        .inspect
        .lock()
        .map_err(|_| "inspect lock poisoned".to_string())?
        .clone();
    let next_config = sanitize_config(&config, Some(&inspect));
    save_config_file(&state.env.config_path, &next_config)?;
    *state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())? = next_config.clone();
    let refreshed = state
        .refreshed_providers
        .lock()
        .map_err(|_| "refreshed providers lock poisoned".to_string())?
        .clone();
    state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?
        .configure(inspect, next_config.clone(), &refreshed);
    Ok(next_config)
}

#[tauri::command]
fn restart_core(config: Value, state: State<'_, GuiState>) -> Result<Value, String> {
    let inspect = state
        .inspect
        .lock()
        .map_err(|_| "inspect lock poisoned".to_string())?
        .clone();
    let next_config = sanitize_config(&config, Some(&inspect));
    save_config_file(&state.env.config_path, &next_config)?;
    *state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())? = next_config.clone();
    let refreshed = state
        .refreshed_providers
        .lock()
        .map_err(|_| "refreshed providers lock poisoned".to_string())?
        .clone();
    let mut manager = state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?;
    manager.configure(inspect, next_config.clone(), &refreshed);
    manager.restart_current(next_config)?;
    Ok(json!({ "ok": true }))
}

#[tauri::command(rename_all = "camelCase")]
fn send_command(
    r#type: String,
    payload: Value,
    session_id: Option<String>,
    state: State<'_, GuiState>,
) -> Result<Value, String> {
    state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?
        .send_command(&r#type, payload, session_id)
}

#[tauri::command]
fn refresh_provider_models(provider: String, state: State<'_, GuiState>) -> Result<Value, String> {
    let provider_name = provider.trim().to_string();
    if provider_name.is_empty() {
        return Err("provider is required".to_string());
    }
    let ready_inspect = state
        .inspect
        .lock()
        .map_err(|_| "inspect lock poisoned".to_string())?
        .clone();
    let ready_config = state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())?
        .clone();

    let refresh_frame = state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?
        .refresh_models(&provider_name)?;

    let mut next_inspect = ready_inspect.clone();
    let all_models = if let Some(frame) = refresh_frame {
        frame_payload(&frame)
            .get("all_models")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
    } else {
        let refreshed = load_inspect_payload(
            &state.env.repo_root,
            &state.env.workspace_root,
            &state.env.engine_launcher,
            &["--refresh-provider".to_string(), provider_name.clone()],
        )?;
        let models = metadata_models(&refreshed);
        next_inspect = refreshed;
        Some(models)
    }
    .ok_or_else(|| format!("refresh for provider '{provider_name}' returned no models"))?;

    set_value_field(
        &mut next_inspect,
        "models",
        Value::Array(
            preserve_provider_order(metadata_models(&ready_inspect), all_models)
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    let next_config = sanitize_config(&ready_config, Some(&next_inspect));
    save_config_file(&state.env.config_path, &next_config)?;
    {
        let mut refreshed = state
            .refreshed_providers
            .lock()
            .map_err(|_| "refreshed providers lock poisoned".to_string())?;
        refreshed.insert(provider_name);
    }
    *state
        .inspect
        .lock()
        .map_err(|_| "inspect lock poisoned".to_string())? = next_inspect.clone();
    *state
        .config
        .lock()
        .map_err(|_| "config lock poisoned".to_string())? = next_config.clone();
    let refreshed = state
        .refreshed_providers
        .lock()
        .map_err(|_| "refreshed providers lock poisoned".to_string())?
        .clone();
    let snapshot = {
        let mut manager = state
            .manager
            .lock()
            .map_err(|_| "manager lock poisoned".to_string())?;
        manager.configure(next_inspect.clone(), next_config.clone(), &refreshed);
        manager.snapshot()
    };
    Ok(gui_metadata(next_inspect, next_config, snapshot))
}

#[tauri::command]
async fn select_working_directory(
    app: AppHandle,
    state: State<'_, GuiState>,
) -> Result<Value, String> {
    let current_workspace = state
        .manager
        .lock()
        .map_err(|_| "manager lock poisoned".to_string())?
        .current_workspace_root();
    let (tx, mut rx) = tauri::async_runtime::channel(1);
    app.dialog()
        .file()
        .set_title("切换工作目录")
        .set_directory(&current_workspace)
        .pick_folder(move |folder| {
            let _ = tx.try_send(folder);
        });
    let folder = rx.recv().await.flatten();
    let Some(folder) = folder else {
        return Ok(json!({ "canceled": true }));
    };
    let path = folder
        .into_path()
        .map_err(|error| format!("failed to resolve selected path: {error}"))?;
    Ok(json!({
        "canceled": false,
        "path": path.to_string_lossy(),
    }))
}

#[tauri::command]
fn set_minimum_content_size(app: AppHandle, size: Value) -> Result<Value, String> {
    let width = normalize_minimum_content_dimension(
        size.get("width").and_then(Value::as_f64),
        MIN_CONTENT_WIDTH,
    );
    let height = normalize_minimum_content_dimension(
        size.get("height").and_then(Value::as_f64),
        MIN_CONTENT_HEIGHT,
    );
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "main window is not available".to_string())?;
    window
        .set_min_size(Some(tauri::LogicalSize::new(width, height)))
        .map_err(|error| error.to_string())?;
    let scale_factor = window.scale_factor().map_err(|error| error.to_string())?;
    let current_size = window.outer_size().map_err(|error| error.to_string())?;
    let current_width = current_size.width as f64 / scale_factor;
    let current_height = current_size.height as f64 / scale_factor;
    if current_width < width || current_height < height {
        window
            .set_size(tauri::Size::Logical(tauri::LogicalSize::new(
                current_width.max(width),
                current_height.max(height),
            )))
            .map_err(|error| error.to_string())?;
    }
    Ok(json!({ "ok": true }))
}

#[tauri::command]
fn save_markdown_export(app: AppHandle, payload: Value) -> Result<Value, String> {
    let markdown = payload
        .get("markdown")
        .and_then(Value::as_str)
        .ok_or_else(|| "invalid markdown export payload".to_string())?;
    let suggested =
        safe_markdown_filename(payload.get("suggested_filename").and_then(Value::as_str));
    let file_path = app
        .dialog()
        .file()
        .set_title("导出 Markdown")
        .set_file_name(&suggested)
        .add_filter("Markdown", &["md"])
        .blocking_save_file();
    let Some(file_path) = file_path else {
        return Ok(json!({ "canceled": true }));
    };
    let markdown_path = file_path
        .into_path()
        .map_err(|error| format!("failed to resolve selected path: {error}"))?;
    let markdown_path = ensure_markdown_extension(markdown_path);
    let parsed_dir = markdown_path
        .parent()
        .ok_or_else(|| "selected export path has no parent directory".to_string())?
        .to_path_buf();
    let stem = markdown_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("hawi-export");
    let reference_dir_name = format!("{stem}-ref");
    let original_ref_dir = payload.get("reference_dir_name").and_then(Value::as_str);
    let markdown = if let Some(original_ref_dir) = original_ref_dir {
        if original_ref_dir != reference_dir_name {
            markdown.replace(original_ref_dir, &reference_dir_name)
        } else {
            markdown.to_string()
        }
    } else {
        markdown.to_string()
    };

    fs::create_dir_all(&parsed_dir)
        .map_err(|error| format!("failed to create export directory: {error}"))?;
    fs::write(&markdown_path, markdown)
        .map_err(|error| format!("failed to write markdown export: {error}"))?;

    let references = payload
        .get("references")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut reference_dir = None;
    if !references.is_empty() {
        let dir = parsed_dir.join(reference_dir_name);
        fs::create_dir_all(&dir)
            .map_err(|error| format!("failed to create reference directory: {error}"))?;
        for reference in references {
            let Some(content) = reference.get("content").and_then(Value::as_str) else {
                continue;
            };
            let filename =
                safe_reference_filename(reference.get("filename").and_then(Value::as_str));
            fs::write(dir.join(filename), content)
                .map_err(|error| format!("failed to write reference file: {error}"))?;
        }
        reference_dir = Some(dir);
    }

    Ok(json!({
        "canceled": false,
        "markdownPath": markdown_path,
        "referenceDir": reference_dir,
    }))
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let env_paths = resolve_env_paths(app.handle())
                .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            ensure_engine_workspace(&env_paths)
                .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            let inspect = load_inspect_payload(
                &env_paths.repo_root,
                &env_paths.workspace_root,
                &env_paths.engine_launcher,
                &[],
            )
            .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
            let mut config = load_config(&env_paths.config_path, &inspect);
            if let Some(argv_model) = parse_arg_value("--model") {
                if metadata_models(&inspect).contains(&argv_model) {
                    set_value_field(&mut config, "modelName", Value::String(argv_model));
                    save_config_file(&env_paths.config_path, &config)
                        .map_err(|error| Box::<dyn std::error::Error>::from(error))?;
                }
            }
            let manager = SessionEngineManager::new_shared(
                app.handle().clone(),
                env_paths.repo_root.clone(),
                env_paths.workspace_root.clone(),
                env_paths.backend_log_path.clone(),
                env_paths.engine_launcher.clone(),
            );
            manager
                .lock()
                .map_err(|_| Box::<dyn std::error::Error>::from("manager lock poisoned"))?
                .configure(inspect.clone(), config.clone(), &HashSet::new());
            app.manage(GuiState {
                env: env_paths,
                inspect: Mutex::new(inspect),
                config: Mutex::new(config),
                manager,
                refreshed_providers: Mutex::new(HashSet::new()),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_metadata,
            save_config,
            restart_core,
            refresh_provider_models,
            send_command,
            select_working_directory,
            set_minimum_content_size,
            save_markdown_export
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}

fn gui_metadata(inspect: Value, config: Value, snapshot: Value) -> Value {
    let mut metadata = Map::new();
    metadata.insert("inspect".to_string(), inspect);
    metadata.insert("config".to_string(), config);
    if let Some(snapshot) = snapshot.as_object() {
        for (key, value) in snapshot {
            metadata.insert(key.clone(), value.clone());
        }
    }
    Value::Object(metadata)
}

fn profile_from_config(config: &Value) -> Value {
    json!({
        "version": 1,
        "modelName": value_string(config, "modelName"),
        "systemPrompt": value_string(config, "systemPrompt"),
        "selectedPlugins": string_list(config.get("selectedPlugins")),
        "pluginConfigs": config.get("pluginConfigs").filter(|value| value.is_object()).cloned().unwrap_or_else(|| json!({})),
        "toolCallPurposeEnabled": config.get("toolCallPurposeEnabled").and_then(Value::as_bool).unwrap_or(true),
        "engineArgs": stable_engine_args(config),
    })
}

fn config_from_profile(profile: &Value, default_config: &Value, metadata: &Value) -> Value {
    sanitize_config(
        &json!({
            "version": 1,
            "modelName": non_empty_string(profile.get("modelName")).unwrap_or_else(|| value_string(default_config, "modelName")),
            "systemPrompt": non_empty_string(profile.get("systemPrompt")).unwrap_or_else(|| value_string(default_config, "systemPrompt")),
            "selectedPlugins": string_list(profile.get("selectedPlugins")),
            "pluginConfigs": profile.get("pluginConfigs").filter(|value| value.is_object()).cloned().unwrap_or_else(|| json!({})),
            "toolCallPurposeEnabled": profile.get("toolCallPurposeEnabled").and_then(Value::as_bool).unwrap_or(true),
            "showDebug": default_config.get("showDebug").and_then(Value::as_bool).unwrap_or(false),
        }),
        Some(metadata),
    )
}

fn launch_profile_from_unknown(value: Option<&Value>) -> Option<Value> {
    let value = value?;
    let model_name = non_empty_string(value.get("modelName"))?;
    let system_prompt = value
        .get("systemPrompt")
        .and_then(Value::as_str)?
        .to_string();
    Some(json!({
        "version": 1,
        "modelName": model_name,
        "systemPrompt": system_prompt,
        "selectedPlugins": string_list(value.get("selectedPlugins")),
        "pluginConfigs": value.get("pluginConfigs").filter(|item| item.is_object()).cloned().unwrap_or_else(|| json!({})),
        "toolCallPurposeEnabled": value.get("toolCallPurposeEnabled").and_then(Value::as_bool).unwrap_or(true),
        "engineArgs": value.get("engineArgs").and_then(Value::as_array).map(|items| {
            items.iter().filter_map(Value::as_str).map(str::to_string).map(Value::String).collect::<Vec<_>>()
        }),
    }))
}

fn stable_engine_args(config: &Value) -> Vec<String> {
    let mut args = vec![
        "--model".to_string(),
        value_string(config, "modelName"),
        "--transport".to_string(),
        "stdio".to_string(),
        "--system-prompt".to_string(),
        value_string(config, "systemPrompt"),
        "--plugins".to_string(),
        string_list(config.get("selectedPlugins")).join(","),
    ];
    args.extend(tool_call_purpose_engine_args(
        config
            .get("toolCallPurposeEnabled")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    ));
    args
}

fn normalize_session_meta(value: &Value) -> Value {
    let session_id = value
        .get("session_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    json!({
        "session_id": session_id,
        "name": value.get("name").and_then(Value::as_str).unwrap_or(&session_id),
        "created_at": value.get("created_at").and_then(Value::as_str).unwrap_or(""),
        "updated_at": value.get("updated_at").and_then(Value::as_str).unwrap_or(""),
        "last_checkpoint_event": value.get("last_checkpoint_event").cloned().unwrap_or(Value::Null),
        "components_present": value.get("components_present").and_then(Value::as_array).cloned().unwrap_or_default(),
        "locked": value.get("locked").and_then(Value::as_bool).unwrap_or(false),
        "lock_owner": value.get("lock_owner").filter(|item| item.is_object()).cloned().unwrap_or(Value::Null),
        "load_state": normalize_load_state(value.get("load_state")),
        "loaded_at": value.get("loaded_at").and_then(Value::as_f64),
        "last_finished_at": value.get("last_finished_at").and_then(Value::as_f64),
        "gui_launch_profile": launch_profile_from_unknown(value.get("gui_launch_profile")),
        "last_cwd": value.get("last_cwd").and_then(Value::as_str),
    })
}

fn workspace_root_from_meta(meta: &Value) -> Option<PathBuf> {
    meta.get("last_cwd")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .and_then(|path| normalize_path(path).ok())
}

fn update_record_from_frame(record: &mut EngineRecord, frame: &Value) {
    let payload = frame_payload(frame);
    match frame_type(frame) {
        Some("core.ready") => {
            if let Some(status) = payload.get("status").and_then(Value::as_object) {
                if let Some(agent_state) = status.get("agent_state").and_then(Value::as_str) {
                    record.agent_state = agent_state.to_string();
                }
                if let Some(runner_state) = status.get("runner_state").and_then(Value::as_str) {
                    record.runner_state = runner_state.to_string();
                }
            }
        }
        Some("core.status") => {
            if let Some(agent_state) = payload.get("agent_state").and_then(Value::as_str) {
                record.agent_state = agent_state.to_string();
            }
            if let Some(runner_state) = payload.get("runner_state").and_then(Value::as_str) {
                record.runner_state = runner_state.to_string();
            }
        }
        Some("run.start") => {
            record.agent_state = "RUNNING".to_string();
            record.has_visible_messages = true;
        }
        Some("run.stop") => {
            record.agent_state = "IDLE".to_string();
            record.runner_state = "IDLE".to_string();
            record.last_finished_at = Some(now_ms());
        }
        _ => {}
    }
}

fn ack_frame(command: &str, payload: Value) -> Value {
    let mut payload = object_or_empty(payload);
    payload.insert("command".to_string(), Value::String(command.to_string()));
    payload.insert("ok".to_string(), Value::Bool(true));
    json!({
        "version": VERSION,
        "type": "ack",
        "payload": payload,
    })
}

fn inject_session_id(frame: Value, session_id: &str) -> Value {
    let mut frame = object_or_empty(frame);
    let mut payload = frame
        .remove("payload")
        .map(object_or_empty)
        .unwrap_or_else(|| Map::new());
    payload
        .entry("session_id".to_string())
        .or_insert_with(|| Value::String(session_id.to_string()));
    frame.insert("payload".to_string(), Value::Object(payload));
    Value::Object(frame)
}

fn frame_with_payload(mut frame: Value, payload: Value) -> Value {
    merge_payload(&mut frame, payload);
    frame
}

fn merge_payload(frame: &mut Value, payload: Value) {
    let mut frame_object = object_or_empty(std::mem::take(frame));
    let mut frame_payload = frame_object
        .remove("payload")
        .map(object_or_empty)
        .unwrap_or_default();
    for (key, value) in object_or_empty(payload) {
        frame_payload.insert(key, value);
    }
    frame_object.insert("payload".to_string(), Value::Object(frame_payload));
    *frame = Value::Object(frame_object);
}

fn frame_payload(frame: &Value) -> Map<String, Value> {
    frame
        .get("payload")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn frame_type(frame: &Value) -> Option<&str> {
    frame.get("type").and_then(Value::as_str)
}

fn frame_id(frame: &Value) -> Option<String> {
    frame.get("id").and_then(Value::as_str).map(str::to_string)
}

fn is_core_frame(value: &Value) -> bool {
    value.get("version").and_then(Value::as_str) == Some(VERSION)
        && value.get("type").and_then(Value::as_str).is_some()
}

fn is_command_response_frame(frame: &Value) -> bool {
    matches!(
        frame_type(frame),
        Some("ack" | "error" | "pong" | "core.status")
    )
}

fn load_state_for_record(record: &EngineRecord) -> &'static str {
    if is_running_agent_state(&record.agent_state) {
        "running"
    } else {
        "loaded"
    }
}

fn is_running_agent_state(value: &str) -> bool {
    value == "RUNNING" || value == "INTERRUPTING"
}

fn emit_readonly_engine_event(app: &AppHandle, channel: String, payload: Value) {
    match channel.as_str() {
        "core:stderr" => {
            let _ = app.emit(
                "core:stderr",
                format!("[readonly] {}", payload.as_str().unwrap_or("")),
            );
        }
        "core:spawn" => {
            let mut next = object_or_empty(payload);
            next.insert("mode".to_string(), Value::String("readonly".to_string()));
            let _ = app.emit("core:spawn", next);
        }
        "core:exit" => {
            let mut next = object_or_empty(payload);
            next.insert("mode".to_string(), Value::String("readonly".to_string()));
            let _ = app.emit("core:exit", next);
        }
        "core:event" => {
            if frame_type(&payload) == Some("error") {
                let _ = app.emit("core:event", payload);
            }
        }
        _ => {
            let _ = app.emit(&channel, payload);
        }
    }
}

fn normalize_load_state(value: Option<&Value>) -> Option<&'static str> {
    match value.and_then(Value::as_str) {
        Some("loaded") => Some("loaded"),
        Some("running") => Some("running"),
        Some("unloaded") => Some("unloaded"),
        _ => None,
    }
}

fn command_timeout(command_type: &str) -> u64 {
    match command_type {
        "compact_context" => COMPACT_COMMAND_TIMEOUT_MS,
        "session_export_markdown" | "session_history" | "session_list" | "session_search" => {
            SESSION_COMMAND_TIMEOUT_MS
        }
        _ => DEFAULT_COMMAND_TIMEOUT_MS,
    }
}

fn is_readonly_command(command_type: &str, payload: &Value) -> bool {
    command_type == "session_search"
        || (command_type == "session_history"
            && payload
                .get("read_only")
                .and_then(Value::as_bool)
                .unwrap_or(false))
}

fn is_missing_session_error(error: &CoreCommandError) -> bool {
    let message = error.message.to_ascii_lowercase();
    let details_class = error
        .details
        .as_ref()
        .and_then(|value| value.get("class"))
        .and_then(Value::as_str);
    message.contains("session not found") || details_class == Some("FileNotFoundError")
}

fn compare_sessions_by_created_at(a: &Value, b: &Value) -> std::cmp::Ordering {
    let left = parse_session_time(a);
    let right = parse_session_time(b);
    right.cmp(&left)
}

fn parse_session_time(value: &Value) -> i64 {
    value
        .get("created_at")
        .or_else(|| value.get("updated_at"))
        .and_then(Value::as_str)
        .and_then(|raw| DateTime::parse_from_rfc3339(raw).ok())
        .map(|date| date.timestamp_millis())
        .unwrap_or(0)
}

fn object_or_empty(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(map) => map,
        _ => Map::new(),
    }
}

fn set_value_field(target: &mut Value, key: &str, value: Value) {
    if !target.is_object() {
        *target = json!({});
    }
    if let Some(object) = target.as_object_mut() {
        object.insert(key.to_string(), value);
    }
}

fn metadata_models(metadata: &Value) -> Vec<String> {
    metadata
        .get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_string(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn non_empty_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn normalize_minimum_content_dimension(value: Option<f64>, fallback: f64) -> f64 {
    value
        .filter(|dimension| dimension.is_finite())
        .map(|dimension| dimension.max(fallback).ceil())
        .unwrap_or(fallback)
}

fn parse_arg_value(name: &str) -> Option<String> {
    let inline_prefix = format!("{name}=");
    let args = env::args().collect::<Vec<_>>();
    for arg in &args {
        if let Some(value) = arg.strip_prefix(&inline_prefix) {
            return Some(value.to_string());
        }
    }
    args.windows(2)
        .find(|items| items[0] == name)
        .map(|items| items[1].clone())
}

fn normalize_path(value: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = value.as_ref();
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| format!("failed to normalize path: {error}"))
    }
}

fn home_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

fn is_same_path_or_child(candidate: &Path, parent: &Path) -> bool {
    let candidate = normalize_path(candidate).unwrap_or_else(|_| candidate.to_path_buf());
    let parent = normalize_path(parent).unwrap_or_else(|_| parent.to_path_buf());
    candidate == parent || candidate.starts_with(parent)
}

fn same_workspace_root(left: &Path, right: &Path) -> bool {
    normalize_path(left).unwrap_or_else(|_| left.to_path_buf())
        == normalize_path(right).unwrap_or_else(|_| right.to_path_buf())
}

fn path_delimiter() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}

fn ensure_markdown_extension(path: PathBuf) -> PathBuf {
    if path.extension().is_some() {
        path
    } else {
        let mut path = path;
        path.set_extension("md");
        path
    }
}

fn safe_markdown_filename(value: Option<&str>) -> String {
    let base = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| Path::new(value).file_name().and_then(|name| name.to_str()))
        .unwrap_or("hawi-export.md");
    let filename = if base.to_ascii_lowercase().ends_with(".md") {
        base.to_string()
    } else {
        format!("{base}.md")
    };
    filename_with_timestamp(&filename)
}

fn safe_reference_filename(value: Option<&str>) -> String {
    let base = value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .and_then(|value| Path::new(value).file_name().and_then(|name| name.to_str()))
        .unwrap_or("reference.txt");
    safe_filename(base)
}

fn filename_with_timestamp(filename: &str) -> String {
    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("hawi-export");
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("md");
    if has_timestamp(stem) {
        filename.to_string()
    } else {
        format!("{stem}-{}.{}", local_timestamp_to_seconds(), ext)
    }
}

fn has_timestamp(value: &str) -> bool {
    value.as_bytes().windows(15).any(|window| {
        window[8] == b'-'
            && window[..8].iter().all(u8::is_ascii_digit)
            && window[9..].iter().all(u8::is_ascii_digit)
    })
}

fn safe_filename(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('-');
        }
    }
    let out = out.trim_start_matches(['.', '-']).to_string();
    if out.is_empty() {
        "session".to_string()
    } else {
        out
    }
}

fn short_session_id(value: &str) -> String {
    if value.len() >= 24 && value.starts_with("session-") {
        return value[8..23].to_string();
    }
    if value.len() <= 8 {
        value.to_string()
    } else {
        value[..8].to_string()
    }
}

fn generate_session_id() -> String {
    format!(
        "session-{}-{}",
        local_timestamp_to_seconds(),
        random_hex_suffix(6)
    )
}

fn random_hex_suffix(len: usize) -> String {
    let mut seed = now_ms() ^ ((std::process::id() as u64) << 16);
    let mut out = String::new();
    while out.len() < len {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        out.push_str(&format!("{:x}", seed & 0xffff));
    }
    out.truncate(len);
    out
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn iso_timestamp(ms: u64) -> String {
    DateTime::<Utc>::from(UNIX_EPOCH + Duration::from_millis(ms)).to_rfc3339()
}

fn local_timestamp_to_seconds() -> String {
    Utc::now().format("%Y%m%d-%H%M%S").to_string()
}

fn base36(mut value: u64) -> String {
    if value == 0 {
        return "0".to_string();
    }
    let mut chars = Vec::new();
    while value > 0 {
        let digit = (value % 36) as u8;
        chars.push(match digit {
            0..=9 => (b'0' + digit) as char,
            _ => (b'a' + digit - 10) as char,
        });
        value /= 36;
    }
    chars.iter().rev().collect()
}
