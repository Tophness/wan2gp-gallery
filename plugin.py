import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import os
import re
from PIL import Image
import gc
import subprocess
import json
import hashlib
import time
import shutil

from .gallery_utils import get_thumbnails_in_batch_windows


class GalleryPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.loaded_once = False

        self.settings = self._load_settings()

        self.THUMB_CACHE_MAX_ENTRIES = self.settings.get("max_cache_entries", 3000)
        self._thumb_cache = {}
        self._scan_cache = {}
        self._disk_cache_initialized = False
        self._thumb_disk_cache_root = None
        self._thumb_disk_dir = None
        self._thumb_index_file = None
        self._thumb_disk_index = {}
        self._thumb_disk_index_dirty = False
        self._thumb_disk_last_save_ts = 0.0

        if self.settings.get("use_disk_cache", False) and self.settings.get("clean_on_startup", False):
            self._ensure_disk_thumb_cache()
            self._prune_thumb_cache()

    def _get_settings_file_path(self):
        plugin_base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(plugin_base, "settings.json")

    def _load_settings(self):
        defaults = {
            "use_disk_cache": False,
            "max_cache_entries": 3000,
            "clean_on_startup": False
        }
        settings_path = self._get_settings_file_path()
        if os.path.isfile(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    defaults.update(data)
            except Exception as e:
                print(f"[GalleryPlugin] Failed to read settings.json: {e}")
        return defaults

    def _save_settings(self):
        settings_path = self._get_settings_file_path()
        try:
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"[GalleryPlugin] Failed to save settings.json: {e}")

    def _get_cache_dir_paths(self):
        plugin_base = os.path.dirname(os.path.abspath(__file__))
        cache_base = os.path.join(plugin_base, ".gallery_cache")
        thumb_dir = os.path.join(cache_base, "thumbs")
        index_file = os.path.join(cache_base, "thumb_index.json")
        return cache_base, thumb_dir, index_file

    def _ensure_disk_thumb_cache(self):
        if self._disk_cache_initialized:
            return
        cache_base, thumb_dir, index_file = self._get_cache_dir_paths()
        try:
            os.makedirs(thumb_dir, exist_ok=True)
        except Exception as e:
            print(f"[GalleryPlugin] Could not create cache dir '{thumb_dir}': {e}")

        self._thumb_disk_cache_root = cache_base
        self._thumb_disk_dir = thumb_dir
        self._thumb_index_file = index_file
        self._thumb_disk_index = {}
        self._thumb_disk_index_dirty = False

        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for p, meta in data.items():
                        if isinstance(meta, dict):
                            key = meta.get("key")
                            fn = meta.get("file")
                            ts = meta.get("ts", 0)
                            if isinstance(p, str) and isinstance(key, (list, tuple)) and len(key) == 2 and isinstance(fn, str):
                                self._thumb_disk_index[p] = {
                                    "key": [int(key[0]), int(key[1])],
                                    "file": fn,
                                    "ts": float(ts) if ts is not None else 0.0
                                }
            except Exception as e:
                print(f"[GalleryPlugin] Could not load thumb cache index: {e}")
                self._thumb_disk_index = {}
        self._disk_cache_initialized = True

    def _thumb_disk_file_name(self, abs_path: str) -> str:
        h = hashlib.sha1(abs_path.encode("utf-8", errors="ignore")).hexdigest()
        return f"{h}.b64"

    def _save_thumb_disk_index(self, force=False):
        if not self.settings.get("use_disk_cache", False):
            return
        self._ensure_disk_thumb_cache()
        if not self._thumb_disk_index_dirty and not force:
            return
        now = time.time()
        if (not force) and (now - self._thumb_disk_last_save_ts < 1.0):
            return
        try:
            if self._thumb_index_file:
                tmp = self._thumb_index_file + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self._thumb_disk_index, f, ensure_ascii=False)
                os.replace(tmp, self._thumb_index_file)
                self._thumb_disk_last_save_ts = now
                self._thumb_disk_index_dirty = False
        except Exception as e:
            print(f"[GalleryPlugin] Could not save thumb cache index: {e}")

    def _disk_thumb_get(self, abs_path: str, sig):
        if not self.settings.get("use_disk_cache", False):
            return None
        self._ensure_disk_thumb_cache()
        if not sig:
            return None
        meta = self._thumb_disk_index.get(abs_path)
        if not meta:
            return None
        cached_key = meta.get("key")
        if not isinstance(cached_key, (list, tuple)) or len(cached_key) != 2:
            return None
        if [int(sig[0]), int(sig[1])] != [int(cached_key[0]), int(cached_key[1])]:
            return None
        fname = meta.get("file")
        if not fname or not self._thumb_disk_dir:
            return None
        fpath = os.path.join(self._thumb_disk_dir, fname)
        try:
            if not os.path.exists(fpath):
                return None
            with open(fpath, "r", encoding="utf-8") as f:
                thumb_b64 = f.read().strip()
            if not thumb_b64:
                return None
            meta["ts"] = time.time()
            self._thumb_disk_index_dirty = True
            return thumb_b64
        except Exception as e:
            print(f"[GalleryPlugin] Could not read cached thumbnail '{fpath}': {e}")
            return None

    def _disk_thumb_put(self, abs_path: str, sig, thumb_b64: str):
        if not self.settings.get("use_disk_cache", False):
            return
        self._ensure_disk_thumb_cache()
        if not sig or not thumb_b64 or not self._thumb_disk_dir:
            return
        try:
            fname = self._thumb_disk_file_name(abs_path)
            fpath = os.path.join(self._thumb_disk_dir, fname)
            tmp = fpath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(thumb_b64)
            os.replace(tmp, fpath)
            self._thumb_disk_index[abs_path] = {
                "key": [int(sig[0]), int(sig[1])],
                "file": fname,
                "ts": time.time(),
            }
            self._thumb_disk_index_dirty = True
        except Exception as e:
            print(f"[GalleryPlugin] Could not write cached thumbnail for '{abs_path}': {e}")

    def _disk_thumb_delete(self, abs_path: str):
        self._ensure_disk_thumb_cache()
        meta = self._thumb_disk_index.pop(abs_path, None)
        if meta:
            self._thumb_disk_index_dirty = True
            fname = meta.get("file")
            if fname and self._thumb_disk_dir:
                fpath = os.path.join(self._thumb_disk_dir, fname)
                try:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                except Exception as e:
                    print(f"[GalleryPlugin] Could not delete cached thumbnail '{fpath}': {e}")

    def _prune_thumb_cache(self):
        max_entries = int(self.settings.get("max_cache_entries", 3000))
        if len(self._thumb_cache) > max_entries:
            items = sorted(self._thumb_cache.items(), key=lambda kv: kv[1].get("ts", 0))
            remove_count = len(self._thumb_cache) - max_entries
            for i in range(remove_count):
                try:
                    p, _ = items[i]
                    self._thumb_cache.pop(p, None)
                except Exception:
                    break

        if self.settings.get("use_disk_cache", False):
            self._ensure_disk_thumb_cache()
            if len(self._thumb_disk_index) > max_entries:
                items = sorted(self._thumb_disk_index.items(), key=lambda kv: (kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0))
                remove_count = len(self._thumb_disk_index) - max_entries
                for i in range(remove_count):
                    try:
                        p, _ = items[i]
                        self._disk_thumb_delete(p)
                    except Exception:
                        break
            self._save_thumb_disk_index(force=False)

    def _thumb_sig_from_path(self, path: str):
        try:
            st = os.stat(path)
            mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
            return (int(mtime_ns), int(st.st_size))
        except Exception:
            return None

    def _get_cache_status_text(self):
        if not self.settings.get("use_disk_cache", False):
            return "*Disk Cache Status:* **Disabled** (Original Windows COM memory thumbnails only)"
        cache_base, thumb_dir, _ = self._get_cache_dir_paths()
        if not os.path.isdir(thumb_dir):
            return "*Disk Cache Status:* **Empty** (0 files)"
        try:
            entries = os.listdir(thumb_dir)
            count = len(entries)
            total_bytes = sum(os.path.getsize(os.path.join(thumb_dir, f)) for f in entries if os.path.isfile(os.path.join(thumb_dir, f)))
            mb = total_bytes / (1024 * 1024)
            return f"*Disk Cache Status:* **Enabled** ({count} cached thumbnails, {mb:.1f} MB in `{cache_base}`)"
        except Exception as e:
            return f"*Disk Cache Status:* Error reading cache: {e}"

    def clear_all_disk_cache(self):
        cache_base, thumb_dir, index_file = self._get_cache_dir_paths()
        try:
            if os.path.isdir(thumb_dir):
                shutil.rmtree(thumb_dir, ignore_errors=True)
                os.makedirs(thumb_dir, exist_ok=True)
            if os.path.isfile(index_file):
                os.remove(index_file)
        except Exception as e:
            print(f"[GalleryPlugin] Error clearing disk cache: {e}")
        self._thumb_cache.clear()
        self._scan_cache.clear()
        self._thumb_disk_index = {}
        self._thumb_disk_index_dirty = False
        self._disk_cache_initialized = False
        return self._get_cache_status_text()

    def setup_ui(self):
        self.add_tab(
            tab_id="gallery_tab",
            label="Gallery",
            component_constructor=self.create_gallery_ui,
            position=1
        )
        self.request_global("server_config")
        self.request_global("has_video_file_extension")
        self.request_global("has_image_file_extension")
        self.request_global("has_audio_file_extension")
        self.request_global("get_settings_from_file")
        self.request_global("get_video_info")
        self.request_global("extract_audio_tracks")
        self.request_global("get_file_creation_date")
        self.request_global("get_video_frame")
        self.request_global("are_model_types_compatible")
        self.request_global("get_model_def")
        self.request_global("get_default_settings")
        self.request_global("add_to_sequence")
        self.request_global("set_model_settings")
        self.request_global("generate_dropdown_model_list")
        self.request_global("get_unique_id")
        self.request_global("args")
        self.request_global("get_lora_local_path")
        self.request_component("main")
        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("model_family")
        self.request_component("model_choice")
        self.request_component("refresh_form_trigger")
        self.request_component("image_start")
        self.request_component("image_end")
        self.request_component("image_prompt_type")
        self.request_component("image_start_row")
        self.request_component("image_end_row")
        self.request_component("image_prompt_type_radio")
        self.request_component("image_prompt_type_endcheckbox")
        self.request_component("plugin_data")
        self.register_data_hook("before_metadata_save", self.add_merge_info_to_metadata)

    def _get_roots(self):
        save_path = os.path.abspath(self.server_config.get("save_path", "outputs"))
        image_save_path = os.path.abspath(self.server_config.get("image_save_path", "outputs"))
        audio_save_path = os.path.abspath(self.server_config.get("audio_save_path", save_path))
        roots = []
        for p in [save_path, image_save_path, audio_save_path]:
            if p and os.path.isdir(p) and p not in roots:
                roots.append(p)
        if not roots:
            roots = [os.path.abspath("outputs")]
        return roots

    def _is_within_roots(self, path: str, roots=None) -> bool:
        ap = os.path.abspath(path)
        roots = roots or self._get_roots()
        for r in roots:
            try:
                if os.path.commonpath([ap, r]) == r:
                    return True
            except Exception:
                pass
        return False

    def create_gallery_ui(self):
        css = """
            #gallery-layout {
                display: flex;
                gap: 16px;
                min-height: 75vh;
                align-items: flex-start;
            }
            #gallery-container {
                flex: 3;
                max-height: 80vh;
                overflow-y: auto;
                border: 1px solid var(--border-color-primary);
                padding: 10px;
                background-color: var(--background-fill-secondary);
                border-radius: 8px;
            }
            #metadata-panel-container {
                flex: 1;
                max-height: 80vh;
                overflow-y: auto;
                border: 1px solid var(--border-color-primary);
                padding: 15px;
                background-color: var(--background-fill-primary);
                border-radius: 8px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .gallery-breadcrumbs {
                display: flex;
                align-items: center;
                gap: 6px;
                padding: 8px 12px;
                margin-bottom: 12px;
                background-color: var(--background-fill-primary);
                border: 1px solid var(--border-color-primary);
                border-radius: 6px;
                font-size: 13px;
                overflow-x: auto;
                white-space: nowrap;
            }
            .breadcrumb-item {
                cursor: pointer;
                color: var(--primary-500);
                font-weight: 500;
                text-decoration: underline;
            }
            .breadcrumb-item:hover {
                color: var(--primary-600);
            }
            .breadcrumb-separator {
                color: var(--body-text-color-subdued);
            }
            .breadcrumb-current {
                font-weight: 600;
                color: var(--body-text-color);
            }

            .gallery-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
                gap: 14px;
            }
            .gallery-item {
                position: relative;
                cursor: pointer;
                border: 2px solid var(--border-color-primary);
                border-radius: 8px;
                overflow: hidden;
                display: flex;
                flex-direction: column;
                background-color: var(--background-fill-primary);
                transition: all 0.18s ease-in-out;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            }
            .gallery-item:hover {
                border-color: var(--primary-400);
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.12);
            }
            .gallery-item.selected {
                border-color: var(--primary-500);
                box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.3);
            }
            .gallery-item-thumbnail {
                width: 100%;
                height: 125px;
                background-color: var(--background-fill-secondary);
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .gallery-item-thumbnail img, .gallery-item-thumbnail video {
                width: 100%;
                height: 100%;
                object-fit: contain;
            }
            .gallery-item-name {
                padding: 6px 8px;
                font-size: 11px;
                font-family: var(--font, system-ui, -apple-system, sans-serif);
                line-height: 1.35;
                text-align: center;
                background-color: var(--background-fill-primary);
                color: var(--body-text-color);
                overflow-wrap: break-word;
                word-break: normal;
                border-top: 1px solid var(--border-color-primary);
                min-height: 2.7em;
                max-height: 3.8em;
                overflow: hidden;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
            }
            .metadata-content {
                width: 100%;
                font-family: var(--font, system-ui, -apple-system, sans-serif);
            }
            .metadata-content .placeholder {
                color: var(--body-text-color-subdued);
                text-align: center;
                margin-top: 20px;
                font-style: italic;
            }
            #video_info {
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
                font-family: var(--font, system-ui, -apple-system, sans-serif);
            }
            #video_info tr {
                border: none !important;
                background: transparent !important;
            }
            #video_info td {
                padding: 5px 6px;
                border: none !important;
                font-size: 12px;
                line-height: 1.4;
                vertical-align: top;
            }
            #video_info td.label-cell {
                text-align: right;
                width: 1%;
                white-space: nowrap;
                color: var(--body-text-color-subdued);
                font-weight: 500;
                padding-right: 12px;
            }
            #video_info td.value-cell {
                text-align: left;
                word-break: break-word;
                overflow-wrap: anywhere;
                color: var(--body-text-color);
            }
            #video_info td.value-cell b {
                color: var(--body-text-color);
                font-weight: 600;
            }
            .frame-preview-header {
                font-size: 12px;
                font-weight: 600;
                text-align: center;
                margin-bottom: 4px;
                color: var(--body-text-color-subdued);
            }
            #custom-button {
                font-size: 13px;
                padding: 3px 8px !important;
            }
            #stop-button:hover {
                background: #ff3333 !important;
            }
            #settings-panel {
                border: 1px solid var(--border-color-primary);
                background-color: var(--background-fill-secondary);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 12px;
            }
            #delete-confirm-box {
                border: 1px solid #ff4444;
                background-color: rgba(255, 68, 68, 0.08);
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 12px;
            }
        """

        js = """
            function() {
                function setGradioInputValue(elem, val) {
                    if (!elem) return;
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLInputElement.prototype, 'value'
                    )?.set;
                    const nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    )?.set;
                    if (elem.tagName === 'INPUT' && nativeInputValueSetter) {
                        nativeInputValueSetter.call(elem, val);
                    } else if (elem.tagName === 'TEXTAREA' && nativeTextAreaValueSetter) {
                        nativeTextAreaValueSetter.call(elem, val);
                    } else {
                        elem.value = val;
                    }
                    elem.dispatchEvent(new Event('input', { bubbles: true }));
                    elem.dispatchEvent(new Event('change', { bubbles: true }));
                }

                window.selectGalleryItem = function(event, element) {
                    if (element.classList.contains('gallery-folder')) return;

                    const gallery = element.closest('.gallery-grid');
                    const selectedFilesInput = document.querySelector('#selected-files-backend input, #selected-files-backend textarea');
                    if (!gallery || !selectedFilesInput) { return; }
                    if (!event.ctrlKey && !event.metaKey) {
                        gallery.querySelectorAll('.gallery-item.selected').forEach(el => {
                            if (el !== element) el.classList.remove('selected');
                        });
                    }
                    element.classList.toggle('selected');
                    const selectedItems = Array.from(gallery.querySelectorAll('.gallery-item.selected'));
                    const selectedPaths = selectedItems.map(el => el.dataset.path);
                    setGradioInputValue(selectedFilesInput, selectedPaths.join('||'));
                };

                window.openGalleryFolderByPath = function(targetPath) {
                    const dirInput = document.querySelector('#current-gallery-dir input, #current-gallery-dir textarea');
                    const selectedFilesInput = document.querySelector('#selected-files-backend input, #selected-files-backend textarea');
                    if (!dirInput) return;
                    setGradioInputValue(dirInput, targetPath || "");
                    if (selectedFilesInput) {
                        setGradioInputValue(selectedFilesInput, "");
                    }
                };

                window.openGalleryFolder = function(event, element) {
                    event.preventDefault();
                    event.stopPropagation();
                    const targetPath = element.dataset.path;
                    window.openGalleryFolderByPath(targetPath);
                };

                function setupVideoFrameSeeker(containerId, sliderId, fps) {
                    const container = document.querySelector(`#${containerId}`);
                    const sliderContainer = document.querySelector(`#${sliderId}`);
                    if (!container || !sliderContainer) return;

                    const video = container.querySelector('video');
                    if (!video) return;

                    let frameTime = (fps > 0) ? 1 / fps : 0;
                    let isSeekingFromSlider = false;
                    let debounceTimer;

                    function updateVideoToFrame(frameNumber) {
                        if (frameTime === 0 || !isFinite(video.duration)) return;
                        const maxFrame = Math.floor(video.duration * fps);
                        const clampedFrame = Math.max(1, Math.min(frameNumber, maxFrame || 1));
                        const targetTime = (clampedFrame - 1) * frameTime;
                        if (Math.abs(video.currentTime - targetTime) > frameTime / 2) {
                            video.currentTime = targetTime;
                        }
                    }

                    video.addEventListener('loadedmetadata', () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (sliderInput) setTimeout(() => updateVideoToFrame(parseInt(sliderInput.value, 10)), 100);
                    }, { once: true });

                    video.addEventListener('timeupdate', () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (!isSeekingFromSlider && frameTime > 0 && sliderInput) {
                            const currentFrame = Math.round(video.currentTime / frameTime) + 1;
                            if (sliderInput.value != currentFrame) {
                                sliderInput.value = currentFrame;
                                const numberInput = sliderContainer.querySelector('input[type="number"]');
                                if (numberInput) numberInput.value = currentFrame;
                            }
                        }
                    });

                    const handleSliderInput = () => {
                        const sliderInput = sliderContainer.querySelector('input[type="range"]');
                        if (sliderInput) {
                            isSeekingFromSlider = true;
                            const frameNumber = parseInt(sliderInput.value, 10);
                            clearTimeout(debounceTimer);
                            debounceTimer = setTimeout(() => {
                                updateVideoToFrame(frameNumber);
                            }, 50);
                        }
                    };

                    const handleInteractionEnd = () => {
                        setTimeout(() => { isSeekingFromSlider = false; }, 150);
                    };

                    sliderContainer.addEventListener('input', handleSliderInput);
                    sliderContainer.addEventListener('mouseup', handleInteractionEnd);
                    sliderContainer.addEventListener('touchend', handleInteractionEnd);
                }

                function setupScopedObserver(observerName, rootSelector, targetSelector, callback) {
                    if (!window.scopedObservers) {
                        window.scopedObservers = new Map();
                    }
                    if (window.scopedObservers.has(observerName)) {
                        return;
                    }

                    const startObserving = () => {
                        const rootElement = document.querySelector(rootSelector);
                        if (!rootElement) {
                            setTimeout(startObserving, 250);
                            return;
                        }

                        const processNode = (node) => {
                            if (node.matches && node.matches(targetSelector)) {
                                callback(node);
                            }
                            if (node.querySelectorAll) {
                                node.querySelectorAll(targetSelector).forEach(callback);
                            }
                        };

                        processNode(rootElement);

                        const observer = new MutationObserver((mutationsList) => {
                            for (const mutation of mutationsList) {
                                if (mutation.type === 'childList') {
                                    mutation.addedNodes.forEach(processNode);
                                }
                            }
                        });

                        observer.observe(rootElement, { childList: true, subtree: true });
                        window.scopedObservers.set(observerName, observer);
                    };

                    startObserving();
                }

                setupScopedObserver(
                    'GalleryVideoSeeker',
                    '#gallery_tab_container',
                    '.video-joiner-player',
                    (playerNode) => {
                        if (!playerNode.dataset.initialized) {
                            const containerId = playerNode.id;
                            const { sliderId, fps } = playerNode.dataset;
                            if (containerId && sliderId && !isNaN(parseFloat(fps))) {
                                setupVideoFrameSeeker(containerId, sliderId, parseFloat(fps));
                                playerNode.dataset.initialized = 'true';
                            }
                        }
                    }
                );
            }
        """

        with gr.Blocks() as gallery_blocks:
            gr.HTML(value=f"<style>{css}</style>")
            gallery_blocks.load(fn=None, js=js)
            with gr.Column(elem_id="gallery_tab_container"):
                with gr.Row():
                    self.refresh_gallery_files_btn = gr.Button("🔄 Refresh Files", scale=2)
                    self.settings_toggle_btn = gr.Button("⚙️ Settings", scale=1)
                    self.delete_files_btn = gr.Button("🗑️ Delete Selected", elem_id="stop-button", scale=2)
                    self.delete_all_btn = gr.Button("⚠️ Delete All", elem_id="stop-button", scale=1)

                with gr.Row(visible=False, elem_id="delete-confirm-box") as self.delete_all_confirm_row:
                    with gr.Column():
                        self.delete_all_msg_md = gr.Markdown("⚠️ **Are you sure you want to delete ALL files in this folder view? This action cannot be undone!**")
                        with gr.Row():
                            self.confirm_delete_all_btn = gr.Button("Yes, Delete All", variant="stop")
                            self.cancel_delete_all_btn = gr.Button("Cancel", variant="secondary")

                self.settings_visible_state = gr.State(False)
                with gr.Column(visible=False, elem_id="settings-panel") as self.settings_panel:
                    gr.Markdown("#### ⚙️ Gallery & Cache Settings")
                    with gr.Row():
                        self.setting_use_disk_cache = gr.Checkbox(
                            label="Enable Thumbnail Disk Cache",
                            value=self.settings.get("use_disk_cache", False),
                            info="Caches base64 thumbnails in plugin/.gallery_cache to speed up reloads. Disable if you prefer zero SSD writes."
                        )
                        self.setting_clean_on_startup = gr.Checkbox(
                            label="Auto-clean Cache on Startup",
                            value=self.settings.get("clean_on_startup", False),
                            info="Prunes and removes orphaned thumbnails when WanGP launches."
                        )
                    with gr.Row():
                        self.setting_max_cache_entries = gr.Slider(
                            label="Max Disk Cache Entries",
                            minimum=500,
                            maximum=10000,
                            step=500,
                            value=self.settings.get("max_cache_entries", 3000),
                            interactive=True
                        )
                        self.clear_cache_btn = gr.Button("🗑️ Clear Disk Cache Now", variant="secondary")
                    self.cache_status_info = gr.Markdown(value=self._get_cache_status_text())

                with gr.Row(elem_id="gallery-layout"):
                    self.gallery_html_output = gr.HTML(
                        value="<div class='gallery-grid'><p class='placeholder'>Click 'Refresh Files' to load gallery.</p></div>",
                        elem_id="gallery-container"
                    )
                    with gr.Column(elem_id="metadata-panel-container"):
                        self.join_videos_btn = gr.Button("Join 2 Selected Videos", interactive=False, visible=False)
                        self.recreate_join_btn = gr.Button("Recreate Join From This Video", visible=False, interactive=False)
                        with gr.Column(visible=False) as self.preview_row:
                            self.video_preview = gr.Video(label="Preview", interactive=True, height=250, visible=False, elem_id="main_video_preview")
                            self.image_preview = gr.Image(label="Preview", interactive=False, height=250, visible=False)
                            self.audio_preview = gr.Audio(label="Preview", interactive=False, visible=False)
                            with gr.Row(visible=False) as self.current_frame_buttons_row:
                                self.use_as_start_btn = gr.Button("⬆️ as Start-Image", variant="primary", elem_id="custom-button")
                                self.use_as_end_btn = gr.Button("as End-Image ⬆️", variant="primary", elem_id="custom-button")
                                self.send_to_generator_settings_btn = gr.Button("Use Settings in Generator", interactive=False, visible=False)
                            with gr.Row(visible=False) as self.frame_preview_row:
                                with gr.Column(scale=1, min_width=90):
                                    gr.Markdown("<div class='frame-preview-header'>First Frame</div>")
                                    self.first_frame_preview = gr.Image(show_label=False, interactive=False, height=135, show_download_button=False, container=False)
                                with gr.Column(scale=1, min_width=90):
                                    gr.Markdown("<div class='frame-preview-header'>Last Frame</div>")
                                    self.last_frame_preview = gr.Image(show_label=False, interactive=False, height=135, show_download_button=False, container=False)

                        self.metadata_panel_output = gr.HTML(value="<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>")
                        with gr.Column(visible=False) as self.merge_info_display:
                            gr.Markdown("--- \n #### Merged From")
                            with gr.Row():
                                with gr.Column():
                                    self.merge_source1_prompt = gr.Markdown(elem_classes="metadata-content")
                                    self.merge_source1_image = gr.Image(interactive=False, show_label=False)
                                with gr.Column():
                                    self.merge_source2_prompt = gr.Markdown(elem_classes="metadata-content")
                                    self.merge_source2_image = gr.Image(interactive=False, show_label=False)
                        with gr.Column(visible=False) as self.join_interface:
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### Video 1 (Provides End Frame)")
                                    self.video1_preview = gr.HTML(label="Video 1 Preview")
                                    self.video1_frame_slider = gr.Slider(label="Frame Number", minimum=1, maximum=100, step=1, interactive=True, elem_id="video1_frame_slider")
                                    self.video1_path = gr.Text(visible=False)
                                    self.video1_info = gr.HTML(label="Video 1 Info")
                                with gr.Column():
                                    gr.Markdown("#### Video 2 (Provides Start Frame)")
                                    self.video2_preview = gr.HTML(label="Video 2 Preview")
                                    self.video2_frame_slider = gr.Slider(label="Frame Number", minimum=1, maximum=100, step=1, interactive=True, elem_id="video2_frame_slider")
                                    self.video2_path = gr.Text(visible=False)
                                    self.video2_info = gr.HTML(label="Video 2 Info")
                            with gr.Row():
                                self.send_to_generator_btn = gr.Button("Send Frames to Generator", variant="primary")
                                self.cancel_join_btn = gr.Button("Cancel")

                self.selected_files_for_backend = gr.Text(label="Selected Files", visible=False, elem_id="selected-files-backend")
                self.current_gallery_dir = gr.Text(label="Current Gallery Dir", visible=False, elem_id="current-gallery-dir")
                self.path_for_settings_loader = gr.Text(label="Path for Settings Loader", visible=False)
                self.current_selected_video_path = gr.Text(visible=False)

        outputs_list = [
            self.gallery_html_output,
            self.selected_files_for_backend,
            self.metadata_panel_output,
            self.join_videos_btn,
            self.recreate_join_btn,
            self.send_to_generator_settings_btn,
            self.preview_row,
            self.video_preview,
            self.image_preview,
            self.audio_preview,
            self.frame_preview_row,
            self.first_frame_preview,
            self.last_frame_preview,
            self.join_interface,
            self.merge_info_display,
            self.current_frame_buttons_row,
            self.current_gallery_dir,
            self.delete_all_confirm_row
        ]
        no_updates = {comp: gr.update() for comp in outputs_list}

        def on_tab_select(current_state, current_dir, evt: gr.SelectData):
            if evt.value == "Gallery" and not self.loaded_once:
                self.loaded_once = True
                return self.list_output_files_as_html(current_state, current_dir)
            return no_updates

        self.main_tabs.select(
            fn=on_tab_select,
            inputs=[self.state, self.current_gallery_dir],
            outputs=outputs_list,
        )

        self.refresh_gallery_files_btn.click(
            fn=self.refresh_gallery_files,
            inputs=[self.state, self.current_gallery_dir],
            outputs=outputs_list
        )

        self.current_gallery_dir.change(
            fn=self.list_output_files_as_html,
            inputs=[self.state, self.current_gallery_dir],
            outputs=outputs_list,
            show_progress="hidden"
        )

        self.delete_files_btn.click(
            fn=self.delete_selected_files,
            inputs=[self.selected_files_for_backend, self.state, self.current_gallery_dir],
            outputs=outputs_list,
            show_progress="hidden"
        )

        self.delete_all_btn.click(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=self.delete_all_confirm_row
        )
        self.cancel_delete_all_btn.click(
            fn=lambda: gr.update(visible=False),
            inputs=None,
            outputs=self.delete_all_confirm_row
        )
        self.confirm_delete_all_btn.click(
            fn=self.delete_all_files_in_folder,
            inputs=[self.state, self.current_gallery_dir],
            outputs=outputs_list,
            show_progress="hidden"
        )

        def toggle_settings_panel(is_visible):
            new_vis = not bool(is_visible)
            return new_vis, gr.update(visible=new_vis), self._get_cache_status_text()

        self.settings_toggle_btn.click(
            fn=toggle_settings_panel,
            inputs=[self.settings_visible_state],
            outputs=[self.settings_visible_state, self.settings_panel, self.cache_status_info]
        )

        def on_use_disk_cache_change(val):
            self.settings["use_disk_cache"] = bool(val)
            self._save_settings()
            return self._get_cache_status_text()

        self.setting_use_disk_cache.change(
            fn=on_use_disk_cache_change,
            inputs=[self.setting_use_disk_cache],
            outputs=[self.cache_status_info]
        )

        def on_clean_startup_change(val):
            self.settings["clean_on_startup"] = bool(val)
            self._save_settings()

        self.setting_clean_on_startup.change(
            fn=on_clean_startup_change,
            inputs=[self.setting_clean_on_startup],
            outputs=None
        )

        def on_max_entries_change(val):
            self.settings["max_cache_entries"] = int(val)
            self.THUMB_CACHE_MAX_ENTRIES = int(val)
            self._save_settings()
            if self.settings.get("use_disk_cache", False):
                self._prune_thumb_cache()
            return self._get_cache_status_text()

        self.setting_max_cache_entries.change(
            fn=on_max_entries_change,
            inputs=[self.setting_max_cache_entries],
            outputs=[self.cache_status_info]
        )

        def on_clear_cache_click():
            status = self.clear_all_disk_cache()
            gr.Info("Disk thumbnail cache has been cleared.")
            return status

        self.clear_cache_btn.click(
            fn=on_clear_cache_click,
            inputs=None,
            outputs=[self.cache_status_info]
        )

        self.selected_files_for_backend.change(
            fn=self.update_metadata_panel_and_buttons,
            inputs=[self.selected_files_for_backend, self.state],
            outputs=[
                self.join_videos_btn, self.send_to_generator_settings_btn, self.metadata_panel_output,
                self.path_for_settings_loader, self.preview_row, self.video_preview, self.image_preview, self.audio_preview,
                self.frame_preview_row, self.first_frame_preview, self.last_frame_preview,
                self.join_interface, self.recreate_join_btn, self.merge_info_display,
                self.merge_source1_prompt, self.merge_source1_image, self.merge_source2_prompt, self.merge_source2_image,
                self.current_frame_buttons_row, self.current_selected_video_path
            ],
            show_progress="hidden"
        )

        self.use_as_start_btn.click(
            fn=self.use_current_frame_as_start,
            inputs=[self.current_selected_video_path],
            outputs=[self.image_start, self.main_tabs, self.image_start_row, self.image_prompt_type_radio],
            js="""(video_path) => {
                const video = document.querySelector('#main_video_preview video');
                const time = video ? video.currentTime : 0;
                return [video_path + '|||' + time];
            }"""
        )

        self.use_as_end_btn.click(
            fn=self.use_current_frame_as_end,
            inputs=[self.current_selected_video_path],
            outputs=[self.image_end, self.main_tabs, self.image_end_row, self.image_prompt_type_endcheckbox],
            js="""(video_path) => {
                const video = document.querySelector('#main_video_preview video');
                const time = video ? video.currentTime : 0;
                return [video_path + '|||' + time];
            }"""
        )

        self.join_videos_btn.click(
            fn=self.show_join_interface,
            inputs=[self.selected_files_for_backend, self.state],
            outputs=[
                self.join_interface, self.preview_row, self.merge_info_display, self.metadata_panel_output,
                self.send_to_generator_settings_btn, self.join_videos_btn, self.recreate_join_btn,
                self.video1_preview, self.video2_preview,
                self.video1_path, self.video2_path, self.video1_frame_slider, self.video2_frame_slider,
                self.video1_info, self.video2_info
            ]
        )

        self.recreate_join_btn.click(
            fn=self.recreate_join_interface,
            inputs=[self.path_for_settings_loader, self.state],
            outputs=[
                self.join_interface, self.preview_row, self.merge_info_display, self.metadata_panel_output,
                self.send_to_generator_settings_btn, self.join_videos_btn, self.recreate_join_btn,
                self.video1_preview, self.video2_preview,
                self.video1_path, self.video2_path, self.video1_frame_slider, self.video2_frame_slider,
                self.video1_info, self.video2_info
            ]
        )

        self.send_to_generator_settings_btn.click(
            fn=self.load_settings_and_frames_from_gallery,
            inputs=[self.state, self.path_for_settings_loader],
            outputs=[self.model_family, self.model_choice, self.main_tabs, self.refresh_form_trigger],
            show_progress="hidden"
        )

        self.send_to_generator_btn.click(
            fn=self.send_selected_frames_to_generator,
            inputs=[self.video1_path, self.video1_frame_slider, self.video2_path, self.video2_frame_slider, self.image_prompt_type],
            outputs=[self.image_start, self.image_end, self.main_tabs, self.join_interface, self.image_prompt_type,
                     self.image_start_row, self.image_end_row, self.image_prompt_type_radio, self.image_prompt_type_endcheckbox,
                     self.plugin_data]
        )

        self.cancel_join_btn.click(
            fn=self.update_metadata_panel_and_buttons,
            inputs=[self.selected_files_for_backend, self.state],
            outputs=[
                self.join_videos_btn, self.send_to_generator_settings_btn, self.metadata_panel_output,
                self.path_for_settings_loader, self.preview_row, self.video_preview, self.image_preview, self.audio_preview,
                self.frame_preview_row, self.first_frame_preview, self.last_frame_preview,
                self.join_interface, self.recreate_join_btn, self.merge_info_display,
                self.merge_source1_prompt, self.merge_source1_image, self.merge_source2_prompt, self.merge_source2_image,
                self.current_frame_buttons_row, self.current_selected_video_path
            ]
        )

        return gallery_blocks

    def use_current_frame_as_start(self, video_path_with_time):
        if not video_path_with_time or '|||' not in video_path_with_time:
            gr.Warning("No video selected or invalid data.")
            return gr.update(), gr.update(), gr.update(), gr.update()
        try:
            video_path, current_time_str = video_path_with_time.split('|||')
            current_time = float(current_time_str)
            fps, _, _, _ = self.get_video_info(video_path)
            frame_number = int(current_time * fps)
            current_frame = self.get_video_frame(video_path, frame_number, return_PIL=True)
            gr.Info(f"Current frame (frame {frame_number + 1}) set as Start-Image.")
            return {
                self.image_start: [(current_frame, "Current Frame")],
                self.main_tabs: gr.Tabs(selected="video_gen"),
                self.image_start_row: gr.Row(visible=True),
                self.image_prompt_type_radio: gr.Radio(value="S")
            }
        except Exception as e:
            print(f"Error in use_current_frame_as_start: {e}")
            gr.Warning(f"Error extracting frame: {e}")
            return gr.update(), gr.update(), gr.update(), gr.update()

    def use_current_frame_as_end(self, video_path_with_time):
        if not video_path_with_time or '|||' not in video_path_with_time:
            gr.Warning("No video selected or invalid data.")
            return gr.update(), gr.update(), gr.update(), gr.update()
        try:
            video_path, current_time_str = video_path_with_time.split('|||')
            current_time = float(current_time_str)
            fps, _, _, _ = self.get_video_info(video_path)
            frame_number = int(current_time * fps)
            current_frame = self.get_video_frame(video_path, frame_number, return_PIL=True)
            gr.Info(f"Current frame (frame {frame_number + 1}) set as End-Image.")
            return {
                self.image_end: [(current_frame, "Current Frame")],
                self.main_tabs: gr.Tabs(selected="video_gen"),
                self.image_end_row: gr.Row(visible=True),
                self.image_prompt_type_endcheckbox: gr.Checkbox(value=True)
            }
        except Exception as e:
            print(f"Error in use_current_frame_as_end: {e}")
            gr.Warning(f"Error extracting frame: {e}")
            return gr.update(), gr.update(), gr.update(), gr.update()

    def delete_selected_files(self, selection_str, current_state, current_dir):
        if not selection_str:
            gr.Warning("No files selected for deletion.")
            return self.list_output_files_as_html(current_state, current_dir)

        file_paths = [p for p in selection_str.split('||') if p]
        deleted_count = 0
        failed_count = 0
        touched_dirs = set()

        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    abs_file = os.path.abspath(file_path)
                    touched_dirs.add(os.path.abspath(os.path.dirname(abs_file)))
                    self._thumb_cache.pop(abs_file, None)
                    self._disk_thumb_delete(abs_file)
                    os.remove(file_path)
                    deleted_count += 1

                    base_path = os.path.splitext(file_path)[0]
                    for ext in ['.txt', '.json', '.metadata']:
                        meta_path = base_path + ext
                        if os.path.exists(meta_path):
                            try:
                                os.remove(meta_path)
                            except Exception:
                                pass
                else:
                    failed_count += 1
            except Exception as e:
                failed_count += 1
                print(f"Error deleting file {file_path}: {e}")

        for d in touched_dirs:
            self._scan_cache.pop(d, None)
        self._save_thumb_disk_index(force=True)

        if deleted_count > 0:
            gr.Info(f"Successfully deleted {deleted_count} file(s).")
        if failed_count > 0:
            gr.Warning(f"Failed to delete {failed_count} file(s).")

        return self.refresh_gallery_files(current_state, current_dir)

    def delete_all_files_in_folder(self, current_state, current_dir=""):
        roots = self._get_roots()
        cur = (current_dir or "").strip()
        cur_abs = os.path.abspath(cur) if cur else ""
        target_dirs = [cur_abs] if cur_abs and self._is_within_roots(cur_abs, roots) else roots

        deleted_count = 0
        failed_count = 0
        touched_dirs = set()

        for d in target_dirs:
            if not os.path.isdir(d):
                continue
            touched_dirs.add(os.path.abspath(d))
            try:
                entries = os.listdir(d)
            except Exception as e:
                print(f"Could not list dir {d}: {e}")
                continue

            for name in entries:
                full_path = os.path.join(d, name)
                if os.path.isfile(full_path) and (
                    self.has_video_file_extension(name)
                    or self.has_image_file_extension(name)
                    or self.has_audio_file_extension(name)
                ):
                    try:
                        abs_file = os.path.abspath(full_path)
                        self._thumb_cache.pop(abs_file, None)
                        self._disk_thumb_delete(abs_file)
                        os.remove(full_path)
                        deleted_count += 1

                        base_path = os.path.splitext(full_path)[0]
                        for ext in ['.txt', '.json', '.metadata']:
                            meta_path = base_path + ext
                            if os.path.exists(meta_path):
                                try:
                                    os.remove(meta_path)
                                except Exception:
                                    pass
                    except Exception as e:
                        failed_count += 1
                        print(f"Could not delete {full_path}: {e}")

        for d in touched_dirs:
            self._scan_cache.pop(d, None)
        self._save_thumb_disk_index(force=True)

        if deleted_count > 0:
            gr.Info(f"Deleted {deleted_count} file(s) in current folder.")
        else:
            gr.Info("No files found to delete.")
        if failed_count > 0:
            gr.Warning(f"Failed to delete {failed_count} file(s).")

        res = self.refresh_gallery_files(current_state, current_dir)
        res[self.delete_all_confirm_row] = gr.update(visible=False)
        return res

    def refresh_gallery_files(self, current_state, current_dir=""):
        if current_dir:
            self._scan_cache.pop(os.path.abspath(current_dir), None)
        else:
            for r in self._get_roots():
                self._scan_cache.pop(r, None)
        return self.list_output_files_as_html(current_state, current_dir)

    def _build_breadcrumbs_html(self, cur_abs, roots):
        if not cur_abs:
            return """
            <div class="gallery-breadcrumbs">
                <span class="breadcrumb-current">🏠 Root (Outputs)</span>
            </div>
            """

        matching_root = None
        for r in roots:
            try:
                if os.path.commonpath([cur_abs, r]) == r:
                    matching_root = r
                    break
            except Exception:
                pass

        if not matching_root:
            matching_root = roots[0] if roots else ""

        root_label = os.path.basename(matching_root) or "Root"
        html_code = f"""<div class="gallery-breadcrumbs">
            <span class="breadcrumb-item" onclick="openGalleryFolderByPath('')">🏠 Root ({root_label})</span>"""

        rel = os.path.relpath(cur_abs, matching_root) if matching_root else ""
        if rel and rel != ".":
            parts = rel.replace("\\", "/").split("/")
            accum = matching_root
            for idx, part in enumerate(parts):
                accum = os.path.join(accum, part)
                is_last = (idx == len(parts) - 1)
                safe_accum = json.dumps(accum, ensure_ascii=False)
                html_code += f""" <span class="breadcrumb-separator">/</span> """
                if is_last:
                    html_code += f"""<span class="breadcrumb-current">{part}</span>"""
                else:
                    html_code += f"""<span class="breadcrumb-item" onclick='openGalleryFolderByPath({safe_accum})'>{part}</span>"""

        html_code += "</div>"
        return html_code

    def list_output_files_as_html(self, current_state, current_dir=""):
        roots = self._get_roots()
        cur = (current_dir or "").strip()
        cur_abs = os.path.abspath(cur) if cur else ""
        if cur_abs and (not os.path.isdir(cur_abs) or not self._is_within_roots(cur_abs, roots)):
            cur_abs = ""

        folder_items = []
        file_items = []
        seen_files = set()
        seen_folders = set()

        def add_folder(folder_path: str, display: str):
            ap = os.path.abspath(folder_path)
            if ap in seen_folders:
                return
            seen_folders.add(ap)
            folder_items.append({"path": ap, "name": display})

        def add_file(file_path: str):
            ap = os.path.abspath(file_path)
            if ap in seen_files:
                return
            seen_files.add(ap)
            file_items.append(ap)

        use_disk_cache = self.settings.get("use_disk_cache", False)

        def scan_dir(dir_path: str):
            if use_disk_cache and dir_path in self._scan_cache:
                cached = self._scan_cache[dir_path]
                for fo in cached.get("folders", []):
                    add_folder(fo["path"], fo["name"])
                for f in cached.get("files", []):
                    add_file(f)
                return

            try:
                entries = os.listdir(dir_path)
            except Exception as e:
                print(f"Could not list dir {dir_path}: {e}")
                return

            curr_folders = []
            curr_files = []

            for name in entries:
                full = os.path.join(dir_path, name)
                if os.path.isdir(full):
                    curr_folders.append({"path": os.path.abspath(full), "name": name})
                    add_folder(full, name)

            for name in entries:
                full = os.path.join(dir_path, name)
                if os.path.isfile(full) and (
                    self.has_video_file_extension(name)
                    or self.has_image_file_extension(name)
                    or self.has_audio_file_extension(name)
                ):
                    abs_f = os.path.abspath(full)
                    curr_files.append(abs_f)
                    add_file(full)

            if use_disk_cache:
                self._scan_cache[dir_path] = {"folders": curr_folders, "files": curr_files}

        if not cur_abs:
            for r in roots:
                scan_dir(r)
        else:
            parent = os.path.abspath(os.path.join(cur_abs, os.pardir))
            if parent and parent != cur_abs and self._is_within_roots(parent, roots):
                add_folder(parent, "⬆️ ..")
            scan_dir(cur_abs)

        folder_items.sort(
            key=lambda x: (
                0 if (x["name"].startswith("⬆️") or x["name"] == "..") else 1,
                x["name"].lower()
            )
        )
        try:
            file_items.sort(key=os.path.getctime, reverse=True)
        except Exception:
            file_items.sort(reverse=True)

        thumb_targets = [
            p for p in file_items
            if self.has_video_file_extension(p) or self.has_image_file_extension(p)
        ]

        if use_disk_cache:
            thumbnails_dict = {}
            to_generate = []
            for p in thumb_targets:
                sig = self._thumb_sig_from_path(p)
                if not sig:
                    continue
                cached = self._thumb_cache.get(p)
                if cached and cached.get("key") == sig and cached.get("thumb"):
                    cached["ts"] = time.time()
                    thumbnails_dict[p] = cached["thumb"]
                    continue
                disk_thumb = self._disk_thumb_get(p, sig)
                if disk_thumb:
                    self._thumb_cache[p] = {"key": sig, "thumb": disk_thumb, "ts": time.time()}
                    thumbnails_dict[p] = disk_thumb
                    continue
                to_generate.append(p)

            if to_generate:
                generated = get_thumbnails_in_batch_windows(to_generate) or {}
                for p in to_generate:
                    thumb = generated.get(p)
                    if thumb:
                        sig = self._thumb_sig_from_path(p)
                        if sig:
                            now = time.time()
                            self._thumb_cache[p] = {"key": sig, "thumb": thumb, "ts": now}
                            self._disk_thumb_put(p, sig, thumb)
                            thumbnails_dict[p] = thumb

            self._prune_thumb_cache()
            self._save_thumb_disk_index(force=False)
        else:
            thumbnails_dict = get_thumbnails_in_batch_windows(thumb_targets) or {}

        breadcrumbs_html = self._build_breadcrumbs_html(cur_abs, roots)
        items_html = ""

        for fo in folder_items:
            fpath = fo["path"]
            display_name = fo["name"]
            safe_path = json.dumps(fpath, ensure_ascii=False)
            items_html += f"""
            <div class="gallery-item gallery-folder" data-path={safe_path} onclick="openGalleryFolder(event, this)" title="Click to open {display_name}">
                <div class="gallery-item-thumbnail" style="display:flex;align-items:center;justify-content:center;font-size:42px;">
                    📁
                </div>
                <div class="gallery-item-name" title="{display_name}">{display_name}</div>
            </div>
            """

        for f in file_items:
            basename = os.path.basename(f)
            display_name = basename
            match = re.search(r'_seed\d+_(.+)\.(mp4|jpg|jpeg|png|webp|wav|mp3|flac|ogg|m4a|aac)$',
                              basename, re.IGNORECASE)
            if match:
                display_name = match.group(1)

            is_video = self.has_video_file_extension(f)
            is_audio = self.has_audio_file_extension(f)
            base64_thumb = thumbnails_dict.get(os.path.abspath(f)) or thumbnails_dict.get(f)

            if is_audio:
                thumbnail_html = """
                    <div style="font-size:42px;line-height:1;display:flex;align-items:center;justify-content:center;height:100%;">
                        🔊
                    </div>
                """
            else:
                thumbnail_html = (
                    f'<img src="data:image/jpeg;base64,{base64_thumb}" alt="thumb">'
                    if base64_thumb else
                    (f'<video muted preload="metadata" src="/gradio_api/file={f}#t=0.5"></video>'
                     if is_video
                     else f'<img src="/gradio_api/file={f}" alt="thumb">')
                )

            safe_path = json.dumps(f, ensure_ascii=False)
            items_html += f"""
            <div class="gallery-item" data-path={safe_path} onclick="selectGalleryItem(event, this)">
                <div class="gallery-item-thumbnail">{thumbnail_html}</div>
                <div class="gallery-item-name" title="{basename}">{display_name}</div>
            </div>
            """

        full_html = f"{breadcrumbs_html}<div class='gallery-grid'>{items_html}</div>"
        clear_metadata_html = "<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>"

        return {
            self.gallery_html_output: full_html,
            self.selected_files_for_backend: "",
            self.metadata_panel_output: clear_metadata_html,
            self.join_videos_btn: gr.Button(visible=False),
            self.recreate_join_btn: gr.Button(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.preview_row: gr.Column(visible=False),
            self.video_preview: gr.Video(value=None, visible=False),
            self.image_preview: gr.Image(value=None, visible=False),
            self.audio_preview: gr.Audio(value=None, visible=False),
            self.frame_preview_row: gr.Row(visible=False),
            self.first_frame_preview: gr.Image(value=None),
            self.last_frame_preview: gr.Image(value=None),
            self.join_interface: gr.Column(visible=False),
            self.merge_info_display: gr.Column(visible=False),
            self.current_frame_buttons_row: gr.Row(visible=False),
            self.current_gallery_dir: cur_abs if cur_abs else "",
            self.delete_all_confirm_row: gr.Row(visible=False)
        }

    def add_merge_info_to_metadata(self, configs, plugin_data, **kwargs):
        if plugin_data and "merge_info" in plugin_data:
            configs["merge_info"] = plugin_data["merge_info"]
        return configs

    def probe_audio_ffprobe(self, file_path: str) -> dict:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if p.returncode != 0 or not p.stdout:
                return {}
            data = json.loads(p.stdout)
            streams = data.get("streams", []) or []
            astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
            fmt = data.get("format", {}) or {}
            duration = None
            if fmt.get("duration") is not None:
                try:
                    duration = float(fmt["duration"])
                except Exception:
                    duration = None

            out = {}
            if duration is not None:
                out["duration_s"] = duration
            if astream:
                if astream.get("codec_name"):
                    out["codec"] = astream.get("codec_name")
                if astream.get("sample_rate"):
                    try:
                        out["sample_rate"] = int(astream.get("sample_rate"))
                    except Exception:
                        out["sample_rate"] = astream.get("sample_rate")
                if astream.get("channels") is not None:
                    out["channels"] = astream.get("channels")
                if astream.get("bit_rate") or fmt.get("bit_rate"):
                    br = astream.get("bit_rate") or fmt.get("bit_rate")
                    try:
                        out["bit_rate"] = int(br)
                    except Exception:
                        out["bit_rate"] = br
            return out
        except Exception as e:
            print(f"ffprobe audio error: {e}")
            return {}

    def _format_lora_name(self, lora: str) -> str:
        if hasattr(self, "get_lora_local_path") and callable(self.get_lora_local_path):
            try:
                res = self.get_lora_local_path(None, lora)
                if res:
                    return str(res)
            except Exception:
                pass
        if not isinstance(lora, str):
            return str(lora)
        lora = lora.strip()
        if lora.startswith("http://") or lora.startswith("https://"):
            parts = lora.split("|")
            base = os.path.basename(parts[0])
            if len(parts) > 1 and parts[1].strip():
                return os.path.join(parts[1].strip(), base)
            return base
        return os.path.basename(lora)

    def _format_loras_display(self, configs: dict) -> str:
        if not isinstance(configs, dict):
            return ""
        loras = configs.get("activated_loras") or configs.get("transformer_loras_filenames") or []
        if isinstance(loras, str):
            loras = [loras]
        if not loras:
            return ""

        raw_mult = configs.get("loras_multipliers") or configs.get("transformer_loras_multipliers") or ""
        mult_list = []
        if isinstance(raw_mult, list):
            mult_list = [str(m) for m in raw_mult]
        elif isinstance(raw_mult, str) and raw_mult.strip():
            for line in raw_mult.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                mult_list.extend(line.replace(",", " ").split())

        mult_list = list(mult_list) + ["1.0"] * max(0, len(loras) - len(mult_list))

        items_html = []
        for lora, mult in zip(loras, mult_list):
            lora_name = self._format_lora_name(lora)
            m_str = str(mult).strip() if mult is not None else ""
            if not m_str:
                m_str = "1.0"
            if not (m_str.startswith("x") or m_str.startswith("X")):
                m_str = f"x{m_str}"
            safe_name = lora_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe_mult = m_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            items_html.append(
                f"<div style='display:flex; justify-content:space-between; align-items:flex-start; gap:8px;'>"
                f"<span style='overflow-wrap:anywhere; word-break:break-word;'>{safe_name}</span>"
                f"<span style='white-space:nowrap; opacity:0.85; font-weight:normal;'>{safe_mult}</span>"
                f"</div>"
            )

        if not items_html:
            return ""
        return f"<div style='display:flex; flex-direction:column; gap:3px;'>{''.join(items_html)}</div>"

    def get_audio_info_html(self, file_path: str, current_state=None) -> str:
        values, labels = [os.path.basename(file_path)], ["File Name"]
        creation_date = str(self.get_file_creation_date(file_path))
        values.append(creation_date[:creation_date.rfind('.')])
        labels.append("Creation Date")

        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            values.append(f"{size_mb:.2f} MB")
            labels.append("File Size")
        except Exception:
            pass

        info = self.probe_audio_ffprobe(file_path)
        if info:
            if "duration_s" in info:
                values.append(f"{info['duration_s']:.2f} s")
                labels.append("Duration")
            if info.get("codec"):
                values.append(info["codec"])
                labels.append("Codec")
            if info.get("sample_rate"):
                values.append(f"{info['sample_rate']} Hz")
                labels.append("Sample Rate")
            if info.get("channels") is not None:
                values.append(info["channels"])
                labels.append("Channels")
            if info.get("bit_rate"):
                try:
                    kbps = int(info["bit_rate"]) / 1000
                    values.append(f"{kbps:.0f} kbps")
                except Exception:
                    values.append(str(info["bit_rate"]))
                labels.append("Bitrate")

        if current_state is not None and hasattr(self, "get_settings_from_file"):
            configs, _, _ = self.get_settings_from_file(current_state, file_path, False, False, False)
            if configs:
                loras_display = self._format_loras_display(configs)
                if loras_display:
                    values.append(loras_display)
                    labels.append("LoRAs")

        rows = [
            f"<TR><TD class='label-cell'>{l}</TD><TD class='value-cell'><B>{v}</B></TD></TR>"
            for l, v in zip(labels, values) if v is not None
        ]
        return f"<TABLE ID=video_info>{''.join(rows)}</TABLE>"

    def get_video_info_html(self, current_state, file_path):
        configs, _, _ = self.get_settings_from_file(current_state, file_path, False, False, False)
        values, labels = [os.path.basename(file_path)], ["File Name"]
        misc_values, misc_labels, pp_values, pp_labels = [], [], [], []
        is_image = self.has_image_file_extension(file_path)
        if is_image:
            width, height = Image.open(file_path).size
            frames_count = fps = 1
            nb_audio_tracks = 0
        else:
            fps, width, height, frames_count = self.get_video_info(file_path)
            nb_audio_tracks = self.extract_audio_tracks(file_path, query_only=True)
        if configs:
            video_model_name = configs.get("type", "Unknown model").split(" - ")[-1]
            misc_values.append(video_model_name)
            misc_labels.append("Model")
            if configs.get("temporal_upsampling"):
                pp_values.append(configs["temporal_upsampling"])
                pp_labels.append("Upsampling")
            if configs.get("film_grain_intensity", 0) > 0:
                pp_values.append(f"Intensity={configs['film_grain_intensity']}, Saturation={configs['film_grain_saturation']}")
                pp_labels.append("Film Grain")

        loras_display = self._format_loras_display(configs) if configs else ""

        if configs is None or "seed" not in configs:
            values.extend(misc_values)
            labels.extend(misc_labels)
            creation_date = str(self.get_file_creation_date(file_path))
            values.append(creation_date[:creation_date.rfind('.')])
            labels.append("Creation Date")
            if is_image:
                values.append(f"{width}x{height}")
                labels.append("Resolution")
            else:
                values.extend([f"{width}x{height}", f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"])
                labels.extend(["Resolution", "Frames"])
            if nb_audio_tracks > 0:
                values.append(nb_audio_tracks)
                labels.append("Nb Audio Tracks")
            values.extend(pp_values)
            labels.extend(pp_labels)
            if loras_display:
                values.append(loras_display)
                labels.append("LoRAs")
        else:
            values.extend(misc_values)
            labels.extend(misc_labels)
            prompt = str(configs.get("prompt", "") or "")[:2048]
            prompt = prompt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            values.append(prompt)
            labels.append("Text Prompt")
            values.extend([
                f"{configs.get('resolution', '')} (real: {width}x{height})",
                configs.get('video_length', 0),
                configs.get('seed', -1),
                configs.get('guidance_scale', 'N/A'),
                configs.get('num_inference_steps', 'N/A')
            ])
            labels.extend(["Resolution", "Video Length", "Seed", "Guidance (CFG)", "Num Inference steps"])
            if loras_display:
                values.append(loras_display)
                labels.append("LoRAs")
            values.extend(pp_values)
            labels.extend(pp_labels)
            creation_date = str(self.get_file_creation_date(file_path))
            values.append(creation_date[:creation_date.rfind('.')])
            labels.append("Creation Date")

        rows = [
            f"<TR><TD class='label-cell'>{l}</TD><TD class='value-cell'><B>{v}</B></TD></TR>"
            for l, v in zip(labels, values) if v is not None
        ]
        return f"<TABLE ID=video_info>{''.join(rows)}</TABLE>"

    def update_metadata_panel_and_buttons(self, selection_str, current_state):
        file_paths = selection_str.split('||') if selection_str else []
        video_files = [f for f in file_paths if self.has_video_file_extension(f)]

        updates = {
            self.join_videos_btn: gr.Button(visible=len(video_files) == 2 and len(file_paths) == 2, interactive=True),
            self.recreate_join_btn: gr.Button(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.path_for_settings_loader: "",
            self.preview_row: gr.Column(visible=False),
            self.video_preview: gr.Video(visible=False, value=None),
            self.image_preview: gr.Image(visible=False, value=None),
            self.audio_preview: gr.Audio(visible=False, value=None),
            self.frame_preview_row: gr.Row(visible=False),
            self.first_frame_preview: gr.Image(value=None),
            self.last_frame_preview: gr.Image(value=None),
            self.join_interface: gr.Column(visible=False),
            self.merge_info_display: gr.Column(visible=False),
            self.metadata_panel_output: gr.HTML(value="<div class='metadata-content'><p class='placeholder'>Select a file to view its metadata.</p></div>", visible=True),
            self.merge_source1_prompt: gr.Markdown(value=""),
            self.merge_source1_image: gr.Image(value=None),
            self.merge_source2_prompt: gr.Markdown(value=""),
            self.merge_source2_image: gr.Image(value=None),
            self.current_frame_buttons_row: gr.Row(visible=False),
            self.current_selected_video_path: ""
        }

        if len(file_paths) == 1:
            file_path = file_paths[0]
            updates[self.path_for_settings_loader] = file_path
            configs, _, _ = self.get_settings_from_file(current_state, file_path, False, False, False)
            updates[self.send_to_generator_settings_btn] = gr.Button(visible=True, interactive=bool(configs))
            if self.has_audio_file_extension(file_path):
                updates[self.metadata_panel_output] = gr.HTML(value=self.get_audio_info_html(file_path, current_state), visible=True)
            else:
                updates[self.metadata_panel_output] = gr.HTML(value=self.get_video_info_html(current_state, file_path), visible=True)

            if configs and "merge_info" in configs:
                merge_info = configs["merge_info"]
                save_path = self.server_config.get("save_path", "outputs")
                image_save_path = self.server_config.get("image_save_path", "outputs")
                vid1_rel, vid2_rel = merge_info['source_video_1']['path'], merge_info['source_video_2']['path']
                vid1_abs = next((p for p in [os.path.join(save_path, vid1_rel), os.path.join(image_save_path, vid1_rel)] if os.path.exists(p)), None)
                vid2_abs = next((p for p in [os.path.join(save_path, vid2_rel), os.path.join(image_save_path, vid2_rel)] if os.path.exists(p)), None)

                if vid1_abs and vid2_abs:
                    updates[self.recreate_join_btn] = gr.Button(visible=True, interactive=True)
                    updates[self.merge_info_display] = gr.Column(visible=True)
                    f1_num, f2_num = merge_info['source_video_1']['frame_used'], merge_info['source_video_2']['frame_used']
                    f1_pil = self.get_video_frame(vid1_abs, f1_num - 1, return_PIL=True)
                    f2_pil = self.get_video_frame(vid2_abs, f2_num - 1, return_PIL=True)
                    c1, _, _ = self.get_settings_from_file(current_state, vid1_abs, False, False, False)
                    c2, _, _ = self.get_settings_from_file(current_state, vid2_abs, False, False, False)
                    p1 = (c1.get('prompt', 'N/A') if c1 else 'N/A')
                    p2 = (c2.get('prompt', 'N/A') if c2 else 'N/A')

                    p1_escaped = p1.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                    p2_escaped = p2.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

                    updates[self.merge_source1_prompt] = f"<b>{vid1_rel} (Frame {f1_num})</b><br>{p1_escaped[:100] + '...' if len(p1_escaped) > 100 else p1_escaped}"
                    updates[self.merge_source1_image] = f1_pil
                    updates[self.merge_source2_prompt] = f"<b>{vid2_rel} (Frame {f2_num})</b><br>{p2_escaped[:100] + '...' if len(p2_escaped) > 100 else p2_escaped}"
                    updates[self.merge_source2_image] = f2_pil
                else:
                    updates[self.preview_row] = gr.Column(visible=True)

            else:
                updates[self.preview_row] = gr.Column(visible=True)

                if self.has_video_file_extension(file_path):
                    updates[self.video_preview] = gr.Video(value=file_path, visible=True)
                    updates[self.image_preview] = gr.Image(visible=False, value=None)
                    updates[self.audio_preview] = gr.Audio(visible=False, value=None)

                    updates[self.current_frame_buttons_row] = gr.Row(visible=True)
                    updates[self.current_selected_video_path] = file_path

                    updates[self.frame_preview_row] = gr.Row(visible=True)
                    first_frame_pil = self.get_video_frame(file_path, 0, return_PIL=True)
                    _, _, _, frame_count = self.get_video_info(file_path)
                    last_frame_pil = (
                        self.get_video_frame(file_path, frame_count - 1, return_PIL=True)
                        if frame_count > 1 else first_frame_pil
                    )

                    updates[self.first_frame_preview] = gr.Image(
                        value=first_frame_pil,
                        visible=True
                    )
                    updates[self.last_frame_preview] = gr.Image(
                        value=last_frame_pil,
                        visible=True
                    )

                elif self.has_image_file_extension(file_path):
                    updates[self.image_preview] = gr.Image(
                        value=Image.open(file_path),
                        label="Image Preview",
                        visible=True
                    )
                    updates[self.video_preview] = gr.Video(visible=False, value=None)
                    updates[self.audio_preview] = gr.Audio(visible=False, value=None)
                    updates[self.current_frame_buttons_row] = gr.Row(visible=False)
                    updates[self.frame_preview_row] = gr.Row(visible=False)
                    updates[self.current_selected_video_path] = ""

                elif self.has_audio_file_extension(file_path):
                    updates[self.audio_preview] = gr.Audio(
                        value=file_path,
                        visible=True
                    )
                    updates[self.video_preview] = gr.Video(visible=False, value=None)
                    updates[self.image_preview] = gr.Image(visible=False, value=None)
                    updates[self.current_frame_buttons_row] = gr.Row(visible=False)
                    updates[self.frame_preview_row] = gr.Row(visible=False)
                    updates[self.current_selected_video_path] = ""

                else:
                    updates[self.video_preview] = gr.Video(visible=False, value=None)
                    updates[self.image_preview] = gr.Image(visible=False, value=None)
                    updates[self.audio_preview] = gr.Audio(visible=False, value=None)
                    updates[self.current_frame_buttons_row] = gr.Row(visible=False)
                    updates[self.frame_preview_row] = gr.Row(visible=False)
                    updates[self.current_selected_video_path] = ""

        elif len(file_paths) > 1:
            updates[self.metadata_panel_output] = gr.HTML(value=f"<div class='metadata-content'><p>{len(file_paths)} items selected.</p></div>", visible=True)

        return updates

    def load_settings_and_frames_from_gallery(self, current_state, file_path):
        if not file_path:
            gr.Warning("No file selected.")
            return gr.update(), gr.update(), gr.update(), gr.update()
        configs, _, _ = self.get_settings_from_file(current_state, file_path, True, True, True)
        if not configs:
            gr.Info("No settings found.")
            return gr.update(), gr.update(), gr.update(), gr.update()
        current_model_type = current_state["model_type"]
        target_model_type = configs.get("model_type", current_model_type)
        if self.are_model_types_compatible(target_model_type, current_model_type):
            target_model_type = current_model_type
        configs["model_type"] = target_model_type
        first_frame, last_frame = None, None
        if self.has_video_file_extension(file_path):
            first_frame = self.get_video_frame(file_path, 0, return_PIL=True)
            _, _, _, frame_count = self.get_video_info(file_path)
            if frame_count > 1:
                last_frame = self.get_video_frame(file_path, frame_count - 1, return_PIL=True)
        elif self.has_image_file_extension(file_path):
            first_frame = Image.open(file_path)
        allowed_prompts = self.get_model_def(target_model_type).get("image_prompt_types_allowed", "")
        configs = {**self.get_default_settings(target_model_type), **configs}
        if first_frame:
            updated_prompts = self.add_to_sequence(configs.get("image_prompt_type", ""), "S") if "S" in allowed_prompts else configs.get("image_prompt_type", "")
            configs["image_start"] = [(first_frame, "First Frame")]
            if last_frame and "E" in allowed_prompts:
                updated_prompts = self.add_to_sequence(updated_prompts, "E")
                configs["image_end"] = [(last_frame, "Last Frame")]
            configs["image_prompt_type"] = updated_prompts
        self.set_model_settings(current_state, target_model_type, configs)
        gr.Info(f"Settings from '{os.path.basename(file_path)}' sent to generator.")
        mf, mbc, mc = (gr.update(), gr.update(), gr.update()) if target_model_type == current_model_type else self.generate_dropdown_model_list(target_model_type)
        return mf, mc, gr.update(selected="video_gen"), self.get_unique_id()

    def show_join_interface(self, selection_str, current_state):
        video_files = [f for f in selection_str.split('||') if self.has_video_file_extension(f)] if selection_str else []
        if len(video_files) != 2:
            gr.Warning("Please select exactly two videos.")
            return {}
        return self.recreate_join_interface(video_files, current_state)

    def recreate_join_interface(self, file_info, current_state):
        if isinstance(file_info, str):
            configs, _, _ = self.get_settings_from_file(current_state, file_info, False, False, False)
            if not (configs and "merge_info" in configs):
                gr.Warning("Could not find merge info in the selected file.")
                return {}
            merge_info = configs["merge_info"]
            save_path = self.server_config.get("save_path", "outputs")
            image_save_path = self.server_config.get("image_save_path", "outputs")
            vid1_rel, vid2_rel = merge_info['source_video_1']['path'], merge_info['source_video_2']['path']
            vid1_abs = next((p for p in [os.path.join(save_path, vid1_rel), os.path.join(image_save_path, vid1_rel)] if os.path.exists(p)), None)
            vid2_abs = next((p for p in [os.path.join(save_path, vid2_rel), os.path.join(image_save_path, vid2_rel)] if os.path.exists(p)), None)
            if not (vid1_abs and vid2_abs):
                gr.Warning("One or both source videos for merging could not be found.")
                return {}
            vid1_path, vid2_path = vid1_abs, vid2_abs
            frame1_num, frame2_num = merge_info['source_video_1']['frame_used'], merge_info['source_video_2']['frame_used']
        elif isinstance(file_info, list) and len(file_info) == 2:
            vid1_path, vid2_path = file_info[0], file_info[1]
            _, _, _, v1_frames = self.get_video_info(vid1_path)
            frame1_num, frame2_num = v1_frames, 1
        else:
            return {}

        server_port_val = int(self.args.server_port) if self.args.server_port != 0 else 7860
        server_name_val = self.args.server_name if self.args.server_name and self.args.server_name != "0.0.0.0" else "127.0.0.1"
        base_url = f"http://{server_name_val}:{server_port_val}"
        v1_fps, _, _, v1_frames = self.get_video_info(vid1_path)
        v2_fps, _, _, v2_frames = self.get_video_info(vid2_path)

        def create_player(container_id, slider_id, path, fps):
            return f'<div id="{container_id}" class="video-joiner-player" data-slider-id="{slider_id}" data-fps="{fps}"><video src="{base_url}/gradio_api/file={path}" style="width:100%;" controls muted preload="metadata"></video></div>'

        player1_html = create_player("video1_player_container", "video1_frame_slider", vid1_path, v1_fps)
        player2_html = create_player("video2_player_container", "video2_frame_slider", vid2_path, v2_fps)

        return {
            self.join_interface: gr.Column(visible=True),
            self.preview_row: gr.Column(visible=False),
            self.merge_info_display: gr.Column(visible=False),
            self.metadata_panel_output: gr.HTML(visible=False),
            self.send_to_generator_settings_btn: gr.Button(visible=False),
            self.join_videos_btn: gr.Button(visible=False),
            self.recreate_join_btn: gr.Button(visible=False),
            self.video1_preview: gr.HTML(value=player1_html),
            self.video2_preview: gr.HTML(value=player2_html),
            self.video1_path: vid1_path,
            self.video2_path: vid2_path,
            self.video1_frame_slider: gr.Slider(maximum=v1_frames, value=frame1_num),
            self.video2_frame_slider: gr.Slider(maximum=v2_frames, value=frame2_num),
            self.video1_info: self.get_video_info_html(current_state, vid1_path),
            self.video2_info: self.get_video_info_html(current_state, vid2_path)
        }

    def send_selected_frames_to_generator(self, vid1_path, frame1_num, vid2_path, frame2_num, current_image_prompt_type):
        frame1 = self.get_video_frame(vid1_path, int(frame1_num) - 1, return_PIL=True)
        frame2 = self.get_video_frame(vid2_path, int(frame2_num) - 1, return_PIL=True)
        gr.Info("Frames sent to Video Generator.")
        updated_image_prompt_type = self.add_to_sequence(current_image_prompt_type, "SE")
        merge_info = {
            "source_video_1": {
                "path": os.path.basename(vid1_path),
                "frame_used": int(frame1_num)
            },
            "source_video_2": {
                "path": os.path.basename(vid2_path),
                "frame_used": int(frame2_num)
            }
        }
        return {
            self.image_start: [(frame1, "Start Frame")],
            self.image_end: [(frame2, "End Frame")],
            self.main_tabs: gr.Tabs(selected="video_gen"),
            self.join_interface: gr.Column(visible=False),
            self.image_prompt_type: updated_image_prompt_type,
            self.image_start_row: gr.Row(visible=True),
            self.image_end_row: gr.Row(visible=True),
            self.image_prompt_type_radio: gr.Radio(value="S"),
            self.image_prompt_type_endcheckbox: gr.Checkbox(value=True),
            self.plugin_data: {"merge_info": merge_info}
        }