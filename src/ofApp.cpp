#include "ofApp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>

namespace {
constexpr std::uint32_t kCacheVersion = 7;
constexpr std::uint32_t kCacheDirectorySignatureVersion = 6;
constexpr std::uint32_t kFrameCacheMagic = 0x43565043; // CVPC
constexpr float kCameraInitialDistance = 0.85f;
constexpr float kCameraInitialMinDistance = 0.24f;
constexpr float kCameraInitialMaxDistance = 3.2f;
constexpr float kCameraFocusDistanceFactor = 1.35f;
constexpr float kCameraDollyDistance = 0.025f;
constexpr float kCameraMaxDistance = 80.0f;
constexpr float kCameraNearClip = 0.0001f;
constexpr float kCameraFarClip = 10000.0f;

std::uint64_t fnv1aAppend(std::uint64_t seed, const void* data, std::size_t size);

std::uint64_t fnv1aAppend(std::uint64_t seed, const void* data, std::size_t size) {
	const auto* bytes = static_cast<const std::uint8_t*>(data);
	for(std::size_t i = 0; i < size; ++i) {
		seed ^= static_cast<std::uint64_t>(bytes[i]);
		seed *= 1099511628211ull;
	}
	return seed;
}

std::uint64_t fnv1aString(std::uint64_t seed, const std::string& value) {
	return fnv1aAppend(seed, value.data(), value.size());
}

template <typename T>
std::uint64_t fnv1aValue(std::uint64_t seed, const T& value) {
	return fnv1aAppend(seed, &value, sizeof(T));
}

template <typename T>
bool readBinary(std::ifstream& input, T& value) {
	input.read(reinterpret_cast<char*>(&value), sizeof(T));
	return static_cast<bool>(input);
}

template <typename T>
bool writeBinary(std::ofstream& output, const T& value) {
	output.write(reinterpret_cast<const char*>(&value), sizeof(T));
	return static_cast<bool>(output);
}

bool fileLooksLocallyAvailable(const std::string& path) {
	struct stat fileStat {};
	if(::stat(path.c_str(), &fileStat) != 0) {
		return false;
	}
	if(!S_ISREG(fileStat.st_mode) || fileStat.st_size <= 0) {
		return false;
	}

	// Dropbox cloud-only placeholders report the logical size but no local blocks.
	return fileStat.st_blocks > 0;
}

bool shouldLogFrameNumber(int frameNumber) {
	return frameNumber < 10 || frameNumber % 100 == 0;
}

bool typeUsedForDisplayNormalization(int type) {
	return type == 0 || type == 2 || type == 3 || type == 4;
}

float packedTypeIntensityValue(int type, float intensity) {
	return static_cast<float>(type) + ofClamp(intensity, 0.0f, 1.0f) * 0.5f;
}
}

void ofApp::setup() {
	ofSetWindowTitle("aq-zoom-000");
	ofSetFrameRate(60);
	ofSetVerticalSync(true);
	ofBackground(0);
	ofEnableDepthTest();
	ofDisableArbTex();
	glEnable(GL_PROGRAM_POINT_SIZE);

	setupGui();
	setupPointShader();

	resetCameraView();

	ofLogNotice() << "[aq-zoom] Setup started";
	dataDirectoryPath = resolveDataDirectoryPath();
	ofLogNotice() << "[aq-zoom] Dataset path: " << dataDirectoryPath;
	frameInfos = loader.discoverFrames(dataDirectoryPath);
	if(frameInfos.empty()) {
		statusMessage = "No snapshot_*.hdf5 files found";
		ofLogError() << "[aq-zoom] " << statusMessage << " in " << dataDirectoryPath;
		return;
	}

	ofLogNotice() << "[aq-zoom] Discovered " << frameInfos.size() << " snapshot frames";
	ofLogNotice() << "[aq-zoom] Using cache version " << kCacheVersion
		<< ", cache particle limit " << cacheParticleLimit;

	if(!preloadAllFrameData()) {
		statusMessage = "Failed to prepare render cache";
		ofLogError() << "[aq-zoom] " << statusMessage;
		return;
	}

	preloadReady = true;
	resetCameraView();
	syncFrameState();
	lastUpdateTime = ofGetElapsedTimef();
	statusMessage = "Ready";
}

void ofApp::update() {
	const double now = ofGetElapsedTimef();
	const double deltaSeconds = std::max(0.0, now - lastUpdateTime);
	lastUpdateTime = now;

	if(!preloadReady || renderFrames.empty()) {
		return;
	}

	constrainCameraDistance();
	finishNextFramePrefetch();
	syncDisplaySettings();

	if(autoRotateParam) {
		rotationDegrees += static_cast<float>(deltaSeconds) * 6.0f;
	}

	if(playing) {
		updatePlaybackPosition(deltaSeconds);
	}
	applyPlaybackPosition(true);
	prepareAheadFrameTexture();
}

void ofApp::draw() {
	ofBackground(0);

	if(preloadReady && frameTexture.isAllocated()) {
		cam.begin();
		ofPushMatrix();
		if(autoRotateParam) {
			ofRotateYDeg(rotationDegrees);
		}
		if(showBoundsParam) {
			drawBounds();
		}
		drawPointCloud();
		ofPopMatrix();
		cam.end();
	}

	if(showGuiParam) {
		ofDisableDepthTest();
		gui.draw();
		ofEnableDepthTest();
	}
	drawHud();
}

void ofApp::exit() {
	waitForPendingPrefetch();
}

void ofApp::keyPressed(int key) {
	if(key == ' ') {
		playing = !playing;
		return;
	}
	if(key == OF_KEY_RIGHT && preloadReady && !renderFrames.empty()) {
		playbackFramePosition += 1.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == OF_KEY_LEFT && preloadReady && !renderFrames.empty()) {
		playbackFramePosition -= 1.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == ']' && preloadReady && !renderFrames.empty()) {
		playbackFramePosition += 100.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == '[' && preloadReady && !renderFrames.empty()) {
		playbackFramePosition -= 100.0;
		applyPlaybackPosition(true);
		return;
	}
	if(key == 'g') {
		showGuiParam = !showGuiParam.get();
		return;
	}
	if(key == 'r') {
		resetCameraView();
		return;
	}
	if(key == 'f') {
		fullResolutionParam = !fullResolutionParam.get();
		return;
	}
	if(key == 'p') {
		performanceModeParam = !performanceModeParam.get();
		return;
	}
	if(key == '0') {
		showGasParam = !showGasParam.get();
		return;
	}
	if(key == '1') {
		showDarkMatterParam = !showDarkMatterParam.get();
		return;
	}
	if(key == '2') {
		showDiskParam = !showDiskParam.get();
		return;
	}
	if(key == '3') {
		showBulgeParam = !showBulgeParam.get();
		return;
	}
	if(key == '4') {
		showStarsParam = !showStarsParam.get();
		return;
	}
}

void ofApp::windowResized(int w, int h) {
	(void)w;
	(void)h;
	gui.setPosition(18, 18);
}

void ofApp::setupGui() {
	guiParams.setName("aq-zoom");
	guiParams.add(pointSizeParam.set("pointSize", 1.0f, 0.6f, 8.0f));
	guiParams.add(sceneScaleParam.set("sceneScale", 1.0f, 0.3f, 8.0f));
	guiParams.add(coreScaleParam.set("viewScale", 1.0f, 0.2f, 8.0f));
	guiParams.add(playbackFpsParam.set("playbackFps", 3.0f, 0.25f, 30.0f));
	guiParams.add(frameStepParam.set("frameStep", 1, 1, 250));
	guiParams.add(temporalSmoothParam.set("smoothMotion", true));
	guiParams.add(brightnessParam.set("brightness", 1.45f, 0.1f, 6.0f));
	guiParams.add(gradientParam.set("gradient", 0.32f, 0.0f, 1.0f));
	guiParams.add(typeColorParam.set("typeColor", 0.65f, 0.0f, 1.0f));
	guiParams.add(maxDrawCountParam.set("drawCap", 300000, 1000, 2000000));
	guiParams.add(fullResolutionParam.set("fullRes", true));
	guiParams.add(autoLodParam.set("autoLOD", true));
	guiParams.add(densityCompensationParam.set("densityBoost", true));
	guiParams.add(depthParticlesParam.set("depthParticles", true));
	guiParams.add(performanceModeParam.set("performance", true));
	guiParams.add(showGasParam.set("gas", true));
	guiParams.add(showDarkMatterParam.set("darkMatter", true));
	guiParams.add(showDiskParam.set("disk", true));
	guiParams.add(showBulgeParam.set("bulge", true));
	guiParams.add(showStarsParam.set("stars", true));
	guiParams.add(showBoundsParam.set("bounds", false));
	guiParams.add(autoRotateParam.set("autoRotate", false));
	guiParams.add(showGuiParam.set("gui", true));
	gui.setup(guiParams);
	gui.setPosition(18, 18);

	previousMaxDrawCount = maxDrawCountParam;
	previousFullResolution = fullResolutionParam;
	previousAutoLod = autoLodParam;
}

void ofApp::setupPointShader() {
	const std::string vertexShader = R"(
		#version 150

		uniform mat4 modelViewMatrix;
		uniform mat4 projectionMatrix;
		uniform sampler2D positionTexture;
		uniform sampler2D nextPositionTexture;
		uniform int textureWidth;
		uniform int displayStride;
		uniform float pointSize;
		uniform float pointDensityScale;
		uniform float sceneScale;
		uniform float viewScale;
		uniform vec3 displayNormalizationCenter;
		uniform float displayNormalizationScale;
		uniform float frameBlend;
		uniform float gradientMix;
		uniform float typeColorMix;
		uniform int showGas;
		uniform int showDarkMatter;
		uniform int showDisk;
		uniform int showBulge;
		uniform int showStars;
		uniform int performanceMode;
		uniform int temporalSmooth;

		in vec4 position;
		out float vType;
		out float vIntensity;
		out float vMotion;
		out float vDepthCue;
		out float vVisible;
		out vec3 vFastColor;
		out float vFastAlpha;

		bool typeEnabled(float typeValue) {
			if(typeValue < 0.5) return showGas != 0;
			if(typeValue < 1.5) return showDarkMatter != 0;
			if(typeValue < 2.5) return showDisk != 0;
			if(typeValue < 3.5) return showBulge != 0;
			if(typeValue < 4.5) return showStars != 0;
			return false;
		}

		vec3 fastBlueGradient(float typeValue, float intensity, float motion) {
			vec3 outerBlue = vec3(0.10, 0.30, 0.78);
			vec3 referenceBlue = vec3(0.22, 0.56, 1.0);
			vec3 coreBlue = vec3(0.62, 0.86, 1.0);
			float signal = clamp(intensity * 0.65 + motion * 0.28, 0.0, 1.0);
			vec3 color = mix(referenceBlue, coreBlue, signal);
			if(typeValue < 1.5 && typeValue >= 0.5) {
				color = mix(outerBlue, referenceBlue, 0.42);
			}
			return color;
		}

		vec3 fastPotentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, clamp(signal, 0.0, 1.0));
		}

		vec3 fastTypeColor(float typeValue, float intensity) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) return mix(gasCold, gasHot, clamp(intensity, 0.0, 1.0));
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		float typeAlpha(float typeValue) {
			if(typeValue < 0.5) return 1.00;
			if(typeValue < 1.5) return 0.24;
			if(typeValue < 2.5) return 0.96;
			if(typeValue < 3.5) return 0.88;
			if(typeValue < 4.5) return 1.00;
			return 0.90;
		}

		void main() {
			int particleIndex = gl_VertexID * displayStride;
			ivec2 texel = ivec2(particleIndex % textureWidth, particleIndex / textureWidth);
			vec4 current = texelFetch(positionTexture, texel, 0);
			vec4 next = current;
			float motionBlend = temporalSmooth != 0 ? frameBlend : 0.0;
			if(motionBlend > 0.0001) {
				next = texelFetch(nextPositionTexture, texel, 0);
			}
			float typeValue = floor(current.a + 0.0001);
			vIntensity = clamp((current.a - typeValue) * 2.0, 0.0, 1.0);
			vVisible = typeEnabled(typeValue) ? 1.0 : 0.0;
			vec3 currentWorld = (current.rgb - displayNormalizationCenter) * displayNormalizationScale;
			vec3 nextWorld = (next.rgb - displayNormalizationCenter) * displayNormalizationScale;
			vec3 world = mix(currentWorld, nextWorld, motionBlend);
			world *= sceneScale * viewScale;
			vec4 view = modelViewMatrix * vec4(world, 1.0);
			float typeSize = typeValue < 0.5 ? 1.20 : (typeValue < 1.5 ? 0.88 : 1.05);
			float perspectiveScale = 1.0;
			float motion = 0.0;
			vDepthCue = 1.0;
			if(performanceMode == 0) {
				float viewDepth = max(-view.z, 0.06);
				perspectiveScale = clamp(0.92 / sqrt(viewDepth), 0.42, 2.35);
				motion = clamp(length(nextWorld - currentWorld) * 18.0, 0.0, 1.0);
				vDepthCue = clamp(1.28 / sqrt(viewDepth), 0.34, 1.35);
			}
			vType = typeValue;
			vMotion = motion;
			vec3 fastBase = fastBlueGradient(typeValue, vIntensity, motion);
			vec3 fastGradient = fastPotentialGradient(vIntensity * 0.45 + motion * 0.20 + 0.22);
			vec3 fastTyped = fastTypeColor(typeValue, vIntensity);
			vFastColor = mix(
				mix(fastBase, fastGradient, clamp(gradientMix, 0.0, 1.0)),
				fastTyped,
				clamp(typeColorMix, 0.0, 1.0)
			);
			vFastAlpha = typeAlpha(typeValue);
			gl_Position = projectionMatrix * view;
			float intensitySize = performanceMode == 0
				? 1.0 + vIntensity * 0.42 + motion * 0.10
				: 1.0 + vIntensity * 0.18;
			gl_PointSize = vVisible > 0.5
				? max(
					pointSize * pointDensityScale * typeSize * perspectiveScale * intensitySize,
					performanceMode == 0 ? 1.1 : 1.6
				)
				: 0.0;
		}
	)";

	const std::string fragmentShader = R"(
		#version 150

		uniform float brightness;
		uniform float densityGain;
		uniform float gradientMix;
		uniform float typeColorMix;
		uniform int fastFullRes;
		in float vType;
		in float vIntensity;
		in float vMotion;
		in float vDepthCue;
		in float vVisible;
		in vec3 vFastColor;
		in float vFastAlpha;
		out vec4 fragColor;

		vec3 blueGradient(float typeValue, float intensity, float motion) {
			vec3 outerBlue = vec3(0.10, 0.30, 0.78);
			vec3 referenceBlue = vec3(0.22, 0.56, 1.0);
			vec3 coreBlue = vec3(0.62, 0.86, 1.0);
			float signal = clamp(intensity * 0.65 + motion * 0.28, 0.0, 1.0);
			vec3 color = mix(referenceBlue, coreBlue, signal);
			if(typeValue < 1.5 && typeValue >= 0.5) {
				color = mix(outerBlue, referenceBlue, 0.42);
			}
			return color;
		}

		vec3 potentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, smoothstep(0.05, 1.0, signal));
		}

		vec3 fastPotentialGradient(float signal) {
			vec3 cyan = vec3(0.10, 0.95, 1.0);
			vec3 warm = vec3(1.0, 0.49, 0.10);
			return mix(cyan, warm, clamp(signal, 0.0, 1.0));
		}

		vec3 typeColor(float typeValue, float intensity, float motion) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) {
				return mix(gasCold, gasHot, smoothstep(0.06, 0.92, max(intensity, motion * 0.55)));
			}
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		vec3 fastTypeColor(float typeValue, float intensity) {
			vec3 gasCold = vec3(0.08, 0.72, 1.00);
			vec3 gasHot = vec3(1.00, 0.43, 0.10);
			vec3 darkMatter = vec3(0.20, 0.26, 0.74);
			vec3 disk = vec3(1.00, 0.70, 0.18);
			vec3 bulge = vec3(1.00, 0.38, 0.50);
			vec3 stars = vec3(1.00, 0.93, 0.68);
			if(typeValue < 0.5) return mix(gasCold, gasHot, clamp(intensity, 0.0, 1.0));
			if(typeValue < 1.5) return darkMatter;
			if(typeValue < 2.5) return disk;
			if(typeValue < 3.5) return bulge;
			if(typeValue < 4.5) return stars;
			return vec3(0.86, 0.92, 1.00);
		}

		float typeAlpha(float typeValue) {
			if(typeValue < 0.5) return 1.00;
			if(typeValue < 1.5) return 0.24;
			if(typeValue < 2.5) return 0.96;
			if(typeValue < 3.5) return 0.88;
			if(typeValue < 4.5) return 1.00;
			return 0.90;
		}

		void main() {
			if(vVisible < 0.5) {
				discard;
			}
			vec2 centered = gl_PointCoord * 2.0 - 1.0;
			float radius2 = dot(centered, centered);
			float glow = exp(-radius2 * 2.6);
			float gain = max(densityGain, 1.0);
			float brightnessGain = sqrt(gain);
			float alphaGain = sqrt(gain);
			if(fastFullRes != 0) {
				fragColor = vec4(
					vFastColor * glow * brightness * brightnessGain * vDepthCue,
					clamp(glow * 0.42 * vFastAlpha * alphaGain, 0.0, 1.0)
				);
				return;
			}
			float gradientSignal = clamp(
				vIntensity * 0.42 +
				vMotion * 0.22 +
				smoothstep(0.34, 1.0, glow) * 0.42,
				0.0,
				1.0
			);
			vec3 baseColor = blueGradient(vType, vIntensity, vMotion);
			vec3 gradColor = potentialGradient(gradientSignal);
			vec3 mixedGradient = mix(baseColor, gradColor, clamp(gradientMix, 0.0, 1.0));
			vec3 typedColor = typeColor(vType, vIntensity, vMotion);
			vec3 color = mix(mixedGradient, typedColor, clamp(typeColorMix, 0.0, 1.0)) * glow;
			float alpha = glow * 0.42 * typeAlpha(vType);
			fragColor = vec4(
				color * brightness * brightnessGain * vDepthCue,
				clamp(alpha * alphaGain, 0.0, 1.0)
			);
		}
	)";

	pointShader.setupShaderFromSource(GL_VERTEX_SHADER, vertexShader);
	pointShader.setupShaderFromSource(GL_FRAGMENT_SHADER, fragmentShader);
	pointShader.bindDefaults();
	pointShader.linkProgram();
}

std::string ofApp::resolveDataDirectoryPath() const {
	if(const char* envPath = std::getenv("AQ_ZOOM_DATA")) {
		const std::string value(envPath);
		if(!value.empty()) {
			return value;
		}
	}

	const std::filesystem::path configPath(ofToDataPath("dataset_path.txt", true));
	std::ifstream config(configPath);
	if(config.is_open()) {
		std::string line;
		std::getline(config, line);
		if(!line.empty()) {
			return line;
		}
	}

	return defaultDataDirectoryPath;
}

bool ofApp::preloadAllFrameData() {
	if(frameInfos.empty()) {
		return false;
	}

	const auto preloadStart = std::chrono::steady_clock::now();
	cacheDirectory = buildCacheDirectory();
	ofLogNotice() << "[aq-zoom] Render cache directory: " << cacheDirectory.string();
	ofLogNotice() << "[aq-zoom] Preloading dataset before playback; HDF5 will not be used during playback";

	if(loadPreprocessedCache(cacheDirectory)) {
		const float preloadMs = std::chrono::duration<float, std::milli>(
			std::chrono::steady_clock::now() - preloadStart
		).count();
		ofLogNotice() << "[aq-zoom] Preload finished from cache in " << ofToString(preloadMs, 1) << " ms";
		return true;
	}

	ofLogNotice() << "[aq-zoom] No usable cache found. Building cache from local HDF5 files";
	std::error_code ec;
	std::filesystem::create_directories(cacheDirectory, ec);
	if(ec) {
		ofLogError() << "[aq-zoom] Failed to create cache directory "
			<< cacheDirectory.string() << ": " << ec.message();
		return false;
	}

	std::size_t skippedOffline = 0;
	std::size_t skippedFailed = 0;
	std::vector<std::uint64_t> slotParticleIds;
	std::unordered_map<std::uint64_t, std::size_t> slotById;
	const std::size_t progressStep = std::max<std::size_t>(1, frameInfos.size() / 200);
	const glm::vec3 rawCenter(0.0f);
	const float rawScale = 1.0f;

	renderFrames.clear();
	renderFrames.reserve(frameInfos.size());

	for(std::size_t frameIndex = 0; frameIndex < frameInfos.size(); ++frameIndex) {
		const auto& info = frameInfos[frameIndex];
		if(frameIndex < 10 || (frameIndex + 1) % progressStep == 0) {
			ofLogNotice() << "[aq-zoom] Loading source frame "
				<< (frameIndex + 1) << "/" << frameInfos.size()
				<< " frame=" << info.frameNumber
				<< " parts=" << std::max<std::size_t>(1, info.partPaths.size());
		}
		bool frameAvailable = fileLooksLocallyAvailable(info.filePath);
		for(const auto& partPath : info.partPaths) {
			if(!fileLooksLocallyAvailable(partPath)) {
				frameAvailable = false;
				break;
			}
		}
		if(!frameAvailable) {
			++skippedOffline;
			continue;
		}

		try {
			const SnapshotFrame frame = loader.loadFrame(info);
			if(frame.particles.empty() || !frame.boundsValid) {
				++skippedFailed;
				continue;
			}

			if(slotParticleIds.empty()) {
				slotParticleIds.reserve(frame.particles.size());
				for(const auto& particle : frame.particles) {
					slotParticleIds.push_back(particle.id);
				}
				std::sort(slotParticleIds.begin(), slotParticleIds.end());
				slotParticleIds.erase(
					std::unique(slotParticleIds.begin(), slotParticleIds.end()),
					slotParticleIds.end()
				);
				if(slotParticleIds.size() > cacheParticleLimit) {
					slotParticleIds.resize(cacheParticleLimit);
				}
				maxParticleCount = slotParticleIds.size();
				textureWidth = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(maxParticleCount))));
				textureHeight = static_cast<int>(
					(maxParticleCount + static_cast<std::size_t>(textureWidth) - 1) /
					static_cast<std::size_t>(textureWidth)
				);
				slotById.reserve(slotParticleIds.size() * 2);
				for(std::size_t slot = 0; slot < slotParticleIds.size(); ++slot) {
					slotById[slotParticleIds[slot]] = slot;
				}
				ofLogNotice() << "[aq-zoom] Fixed ParticleID slots: "
					<< slotParticleIds.size() << " from frame " << info.frameNumber;
				ofLogNotice() << "[aq-zoom] Cache uses raw coordinates with display normalization; texture "
					<< textureWidth << "x" << textureHeight;
			}

			std::vector<RenderParticle> particles = buildRenderParticles(
				frame,
				rawCenter,
				rawScale,
				slotById,
				maxParticleCount
			);
			if(particles.empty()) {
				++skippedFailed;
				continue;
			}

			CachedFrameInfo cachedInfo;
			cachedInfo.sourceIndex = frameIndex;
			cachedInfo.frameNumber = frame.frameNumber;
			cachedInfo.time = frame.time;
			cachedInfo.particleCount = particles.size();
			for(const auto& particle : particles) {
				const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
				if(type >= 0 && type < static_cast<int>(cachedInfo.typeCounts.size())) {
					++cachedInfo.typeCounts[static_cast<std::size_t>(type)];
				}
			}

			renderFrames.push_back(cachedInfo);
			if(!writeCachedFrameData(renderFrames.size() - 1, particles)) {
				ofLogError() << "[aq-zoom] Failed to write cache for frame " << frame.frameNumber;
				renderFrames.pop_back();
				return false;
			}
		} catch(const std::exception& e) {
			++skippedFailed;
			ofLogWarning() << "[aq-zoom] Preload cache skip frame " << info.frameNumber
				<< ": " << e.what();
		}

		if((frameIndex + 1) % progressStep == 0 || frameIndex + 1 == frameInfos.size()) {
			ofLogNotice() << "[aq-zoom] Preload fixed-ID cache phase: "
				<< (frameIndex + 1) << "/" << frameInfos.size()
				<< " cached=" << renderFrames.size()
				<< " offline=" << skippedOffline
				<< " failed=" << skippedFailed;
		}
	}

	if(renderFrames.empty()) {
		ofLogError() << "[aq-zoom] Cache build produced no frames";
		return false;
	}

	if(!writePreprocessedCache(cacheDirectory)) {
		return false;
	}

	if(!computeDisplayNormalizationFromCache()) {
		ofLogWarning() << "[aq-zoom] Failed to compute display normalization; using cached coordinates as-is";
	}
	warmInitialRamCache(0);
	const bool loaded = loadRenderFrame(0);
	const float preloadMs = std::chrono::duration<float, std::milli>(
		std::chrono::steady_clock::now() - preloadStart
	).count();
	ofLogNotice() << "[aq-zoom] Preload finished in " << ofToString(preloadMs, 1)
		<< " ms; cached frames=" << renderFrames.size()
		<< ", skipped offline=" << skippedOffline
		<< ", skipped failed=" << skippedFailed;
	return loaded;
}

std::filesystem::path ofApp::buildCacheDirectory() const {
	std::uint64_t signature = 14695981039346656037ull;
	signature = fnv1aString(signature, dataDirectoryPath);
	signature = fnv1aValue(signature, kCacheDirectorySignatureVersion);
	signature = fnv1aValue(signature, cacheParticleLimit);
	signature = fnv1aValue(signature, frameInfos.size());

	for(const auto& frame : frameInfos) {
		signature = fnv1aValue(signature, frame.frameNumber);
		signature = fnv1aString(signature, frame.filePath);
		for(const auto& partPath : frame.partPaths) {
			signature = fnv1aString(signature, partPath);
			std::error_code ec;
			const std::filesystem::path framePath(partPath);
			if(std::filesystem::exists(framePath, ec)) {
				const auto size = std::filesystem::file_size(framePath, ec);
				if(!ec) {
					signature = fnv1aValue(signature, static_cast<std::uint64_t>(size));
				}
				const auto writeTime = std::filesystem::last_write_time(framePath, ec);
				if(!ec) {
					signature = fnv1aValue(signature, writeTime.time_since_epoch().count());
				}
			}
		}
	}

	std::ostringstream name;
	name << "dataset-" << std::hex << std::setw(16) << std::setfill('0') << signature
		<< "-limit-" << std::dec << cacheParticleLimit;

	if(const char* envPath = std::getenv("COLLIDE_VIS_CACHE")) {
		const std::string value(envPath);
		if(!value.empty()) {
			return std::filesystem::path(value) / name.str();
		}
	}

	const std::string biwinPrefix = "/Volumes/BIWIN/";
	if(dataDirectoryPath.rfind(biwinPrefix, 0) == 0) {
		return std::filesystem::path("/Volumes/BIWIN/aq-zoom-000-cache") / name.str();
	}

	const std::filesystem::path localCacheRoot(ofToDataPath("aq-zoom-000-cache", true));
	const std::filesystem::path computedCacheDir = localCacheRoot / name.str();
	std::error_code ec;
	if(std::filesystem::exists(computedCacheDir / "manifest.txt", ec)) {
		return computedCacheDir;
	}

	const std::string limitSuffix = "-limit-" + ofToString(cacheParticleLimit);
	std::filesystem::path fallbackCacheDir;
	std::size_t fallbackCount = 0;
	if(std::filesystem::exists(localCacheRoot, ec)) {
		for(const auto& entry : std::filesystem::directory_iterator(localCacheRoot, ec)) {
			if(ec) {
				break;
			}
			if(!entry.is_directory(ec)) {
				continue;
			}
			const std::string dirname = entry.path().filename().string();
			if(dirname.rfind("dataset-", 0) != 0 ||
			   dirname.size() < limitSuffix.size() ||
			   dirname.compare(dirname.size() - limitSuffix.size(), limitSuffix.size(), limitSuffix) != 0) {
				continue;
			}

			std::ifstream manifest(entry.path() / "manifest.txt");
			std::uint32_t version = 0;
			std::size_t sourceFrameCount = 0;
			std::size_t cachedFrameCount = 0;
			int manifestTextureWidth = 0;
			int manifestTextureHeight = 0;
			glm::vec3 manifestMinBounds(0.0f);
			glm::vec3 manifestMaxBounds(0.0f);
			std::size_t manifestMaxParticleCount = 0;
			std::size_t manifestCacheParticleLimit = 0;
			manifest >> version;
			manifest >> sourceFrameCount;
			manifest >> cachedFrameCount;
			manifest >> manifestTextureWidth >> manifestTextureHeight;
			manifest >> manifestMinBounds.x >> manifestMinBounds.y >> manifestMinBounds.z;
			manifest >> manifestMaxBounds.x >> manifestMaxBounds.y >> manifestMaxBounds.z;
			manifest >> manifestMaxParticleCount;
			manifest >> manifestCacheParticleLimit;
			if(manifest &&
			   version == kCacheVersion &&
			   cachedFrameCount > 0 &&
			   manifestCacheParticleLimit == cacheParticleLimit) {
				fallbackCacheDir = entry.path();
				++fallbackCount;
			}
		}
	}

	if(fallbackCount == 1) {
		ofLogNotice() << "[aq-zoom] Using existing local render cache "
			<< fallbackCacheDir.string()
			<< " because no cache exists for the current dataset path signature";
		return fallbackCacheDir;
	}

	return computedCacheDir;
}

bool ofApp::loadPreprocessedCache(const std::filesystem::path& cacheDir) {
	const std::filesystem::path manifestPath = cacheDir / "manifest.txt";
	if(!std::filesystem::exists(manifestPath)) {
		return false;
	}

	std::ifstream manifest(manifestPath);
	if(!manifest.is_open()) {
		return false;
	}

	std::uint32_t version = 0;
	std::size_t sourceFrameCount = 0;
	std::size_t cachedFrameCount = 0;
	std::size_t manifestMaxParticleCount = 0;
	std::size_t manifestCacheParticleLimit = 0;
	manifest >> version;
	manifest >> sourceFrameCount;
	manifest >> cachedFrameCount;
	manifest >> textureWidth >> textureHeight;
	manifest >> worldMin.x >> worldMin.y >> worldMin.z;
	manifest >> worldMax.x >> worldMax.y >> worldMax.z;
	manifest >> manifestMaxParticleCount;
	manifest >> manifestCacheParticleLimit;
	if(!manifest ||
	   version != kCacheVersion ||
	   manifestCacheParticleLimit != cacheParticleLimit ||
	   cachedFrameCount == 0) {
		return false;
	}
	if(sourceFrameCount != frameInfos.size()) {
		ofLogWarning() << "[aq-zoom] Cache source frame count " << sourceFrameCount
			<< " differs from current local snapshot count " << frameInfos.size()
			<< "; using cached frames without rebuilding from HDF5";
	}

	maxParticleCount = manifestMaxParticleCount;
	renderFrames.clear();
	renderFrames.reserve(cachedFrameCount);
	for(std::size_t i = 0; i < cachedFrameCount; ++i) {
		CachedFrameInfo info;
		manifest >> info.sourceIndex;
		manifest >> info.frameNumber;
		manifest >> info.time;
		manifest >> info.particleCount;
		for(auto& typeCount : info.typeCounts) {
			manifest >> typeCount;
		}
		if(!manifest || info.particleCount == 0) {
			return false;
		}
		renderFrames.push_back(info);
	}

	const std::size_t progressStep = std::max<std::size_t>(1, renderFrames.size() / 10);
	for(std::size_t frameIndex = 0; frameIndex < renderFrames.size(); ++frameIndex) {
		if(!loadCachedFrameData(frameIndex, nullptr)) {
			ofLogWarning() << "[aq-zoom] Cache manifest exists, but frame cache is missing/corrupt at index "
				<< frameIndex;
			renderFrames.clear();
			return false;
		}
		if((frameIndex + 1) % progressStep == 0 || frameIndex + 1 == renderFrames.size()) {
			ofLogNotice() << "[aq-zoom] Cache verify phase: "
				<< (frameIndex + 1) << "/" << renderFrames.size() << " frames";
		}
	}

	ofLogNotice() << "[aq-zoom] Loaded render cache manifest from " << cacheDir.string()
		<< " with " << renderFrames.size() << " cached frames";
	if(!computeDisplayNormalizationFromCache()) {
		ofLogWarning() << "[aq-zoom] Failed to compute display normalization; using cached coordinates as-is";
	}
	warmInitialRamCache(0);
	return loadRenderFrame(0);
}

bool ofApp::writePreprocessedCache(const std::filesystem::path& cacheDir) const {
	const std::filesystem::path manifestPath = cacheDir / "manifest.txt";
	const std::filesystem::path tempPath = cacheDir / "manifest.tmp";
	std::ofstream manifest(tempPath, std::ios::trunc);
	if(!manifest.is_open()) {
		ofLogWarning() << "[aq-zoom] Failed to write cache manifest: " << tempPath.string();
		return false;
	}

	manifest << kCacheVersion << '\n';
	manifest << frameInfos.size() << '\n';
	manifest << renderFrames.size() << '\n';
	manifest << textureWidth << ' ' << textureHeight << '\n';
	manifest << worldMin.x << ' ' << worldMin.y << ' ' << worldMin.z << '\n';
	manifest << worldMax.x << ' ' << worldMax.y << ' ' << worldMax.z << '\n';
	manifest << maxParticleCount << '\n';
	manifest << cacheParticleLimit << '\n';
	for(const auto& frame : renderFrames) {
		manifest << frame.sourceIndex << ' '
			<< frame.frameNumber << ' '
			<< std::setprecision(17) << frame.time << ' '
			<< frame.particleCount;
		for(std::size_t typeCount : frame.typeCounts) {
			manifest << ' ' << typeCount;
		}
		manifest << '\n';
	}
	manifest.close();
	if(!manifest) {
		ofLogWarning() << "[aq-zoom] Failed to finalize cache manifest: " << tempPath.string();
		return false;
	}

	std::error_code ec;
	std::filesystem::rename(tempPath, manifestPath, ec);
	if(ec) {
		std::filesystem::remove(manifestPath, ec);
		ec.clear();
		std::filesystem::rename(tempPath, manifestPath, ec);
	}
	if(ec) {
		ofLogWarning() << "[aq-zoom] Failed to move cache manifest into place: " << ec.message();
		return false;
	}

	ofLogNotice() << "[aq-zoom] Wrote render cache manifest to " << manifestPath.string();
	return true;
}

bool ofApp::loadCachedFrameData(std::size_t frameIndex, std::vector<RenderParticle>* particles) const {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	const auto& info = renderFrames[frameIndex];
	const std::filesystem::path framePath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".bin");
	std::ifstream input(framePath, std::ios::binary);
	if(!input.is_open()) {
		return false;
	}

	std::uint32_t magic = 0;
	std::uint32_t version = 0;
	int frameNumber = -1;
	double time = 0.0;
	std::uint32_t particleCount = 0;
	if(!readBinary(input, magic) ||
	   !readBinary(input, version) ||
	   !readBinary(input, frameNumber) ||
	   !readBinary(input, time) ||
	   !readBinary(input, particleCount)) {
		return false;
	}

	std::array<std::uint64_t, 6> typeCounts {};
	for(auto& typeCount : typeCounts) {
		if(!readBinary(input, typeCount)) {
			return false;
		}
	}

	if(magic != kFrameCacheMagic ||
	   version != kCacheVersion ||
	   frameNumber != info.frameNumber ||
	   particleCount != info.particleCount ||
	   particleCount > cacheParticleLimit + 1024) {
		return false;
	}

	if(particles == nullptr) {
		return true;
	}

	particles->clear();
	particles->resize(static_cast<std::size_t>(particleCount));
	for(auto& particle : *particles) {
		float payload[4] = {0.0f, 0.0f, 0.0f, 0.0f};
		input.read(reinterpret_cast<char*>(payload), sizeof(payload));
		if(!input) {
			particles->clear();
			return false;
		}
		particle.position = glm::vec3(payload[0], payload[1], payload[2]);
		particle.nextPosition = particle.position;
		particle.typeIntensity = payload[3];
	}

	return true;
}

bool ofApp::writeCachedFrameData(std::size_t frameIndex, const std::vector<RenderParticle>& particles) const {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}

	std::error_code ec;
	std::filesystem::create_directories(cacheDirectory, ec);
	if(ec) {
		ofLogWarning() << "[aq-zoom] Failed to create cache directory: " << cacheDirectory.string();
		return false;
	}

	const auto& info = renderFrames[frameIndex];
	const std::filesystem::path tempPath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".tmp");
	const std::filesystem::path framePath =
		cacheDirectory / ("frame-" + ofToString(info.frameNumber) + ".bin");
	std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
	if(!output.is_open()) {
		ofLogWarning() << "[aq-zoom] Failed to open frame cache for writing: " << tempPath.string();
		return false;
	}

	const std::uint32_t magic = kFrameCacheMagic;
	const std::uint32_t version = kCacheVersion;
	const std::uint32_t particleCount = static_cast<std::uint32_t>(particles.size());
	if(!writeBinary(output, magic) ||
	   !writeBinary(output, version) ||
	   !writeBinary(output, info.frameNumber) ||
	   !writeBinary(output, info.time) ||
	   !writeBinary(output, particleCount)) {
		return false;
	}
	for(std::size_t typeCount : info.typeCounts) {
		const std::uint64_t value = static_cast<std::uint64_t>(typeCount);
		if(!writeBinary(output, value)) {
			return false;
		}
	}
	for(const auto& particle : particles) {
		const float payload[4] = {
			particle.position.x,
			particle.position.y,
			particle.position.z,
			particle.typeIntensity
		};
		output.write(reinterpret_cast<const char*>(payload), sizeof(payload));
		if(!output) {
			return false;
		}
	}
	output.close();
	if(!output) {
		std::filesystem::remove(tempPath, ec);
		return false;
	}

	std::filesystem::rename(tempPath, framePath, ec);
	if(ec) {
		std::filesystem::remove(framePath, ec);
		ec.clear();
		std::filesystem::rename(tempPath, framePath, ec);
	}
	return !ec;
}

bool ofApp::computeDisplayNormalizationFromCache() {
	if(renderFrames.empty()) {
		return false;
	}

	const std::size_t sampleFrameCount = std::min<std::size_t>(15, renderFrames.size());
	const std::size_t targetSamplesPerFrame = 18000;
	std::vector<glm::vec3> samples;
	samples.reserve(sampleFrameCount * targetSamplesPerFrame);

	for(std::size_t i = 0; i < sampleFrameCount; ++i) {
		const std::size_t frameIndex = sampleFrameCount <= 1
			? 0
			: (i * (renderFrames.size() - 1)) / (sampleFrameCount - 1);
		std::vector<RenderParticle> particles;
		if(!loadCachedFrameData(frameIndex, &particles) || particles.empty()) {
			continue;
		}

		const std::size_t stride = std::max<std::size_t>(1, particles.size() / targetSamplesPerFrame);
		for(std::size_t particleIndex = 0; particleIndex < particles.size(); particleIndex += stride) {
			const auto& particle = particles[particleIndex];
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(!typeUsedForDisplayNormalization(type)) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			samples.push_back(position);
		}
	}

	if(samples.size() < 32) {
		return false;
	}

	auto medianComponent = [&samples](int component) {
		std::vector<float> values;
		values.reserve(samples.size());
		for(const auto& sample : samples) {
			values.push_back(sample[component]);
		}
		const std::size_t middle = values.size() / 2;
		std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
		return values[middle];
	};

	const glm::vec3 center(
		medianComponent(0),
		medianComponent(1),
		medianComponent(2)
	);

	std::vector<float> radii;
	radii.reserve(samples.size());
	for(const auto& sample : samples) {
		radii.push_back(glm::length(sample - center));
	}
	std::sort(radii.begin(), radii.end());
	const auto quantile = [&radii](float value) {
		const std::size_t index = std::min<std::size_t>(
			radii.size() - 1,
			static_cast<std::size_t>(std::floor(value * static_cast<float>(radii.size() - 1)))
		);
		return radii[index];
	};

	const float radius50 = quantile(0.50f);
	const float radius90 = quantile(0.90f);
	const float radius95 = quantile(0.95f);
	const float radius99 = quantile(0.99f);
	const float normalizationRadius = std::max(radius95, std::numeric_limits<float>::epsilon());

	displayNormalizationCenter = center;
	displayNormalizationScale = 0.95f / normalizationRadius;
	displayNormalizationReady = true;
	ofLogNotice() << "[aq-zoom] Display normalization from cache: samples=" << samples.size()
		<< " center=(" << displayNormalizationCenter.x
		<< ", " << displayNormalizationCenter.y
		<< ", " << displayNormalizationCenter.z
		<< ") r50=" << radius50
		<< " r90=" << radius90
		<< " r95=" << radius95
		<< " r99=" << radius99
		<< " scale=" << displayNormalizationScale;
	return true;
}

void ofApp::warmInitialRamCache(std::size_t startFrameIndex) {
	if(renderFrames.empty() || maxRamCachedFrames == 0) {
		return;
	}

	const std::size_t warmFrameCount = std::min<std::size_t>(24, std::min(maxRamCachedFrames, renderFrames.size()));
	ofLogNotice() << "[aq-zoom] Warming RAM frame cache: " << warmFrameCount
		<< " frame(s), about " << ofToString(
			static_cast<double>(warmFrameCount) *
			static_cast<double>(maxParticleCount) *
			static_cast<double>(sizeof(RenderParticle)) /
			(1024.0 * 1024.0),
			0
		) << " MiB";

	const auto warmStart = std::chrono::steady_clock::now();
	for(std::size_t i = 0; i < warmFrameCount; ++i) {
		const std::size_t frameIndex = (startFrameIndex + i) % renderFrames.size();
		if(findRamCachedFrame(frameIndex) != nullptr) {
			continue;
		}
		std::vector<RenderParticle> particles;
		if(!loadCachedFrameData(frameIndex, &particles) || particles.empty()) {
			ofLogWarning() << "[aq-zoom] RAM warm skip frame index " << frameIndex;
			continue;
		}
		storeRamCachedFrame(frameIndex, std::move(particles));
		if((i + 1) % 6 == 0 || i + 1 == warmFrameCount) {
			ofLogNotice() << "[aq-zoom] RAM warm phase: "
				<< (i + 1) << "/" << warmFrameCount
				<< " cached=" << ramFrameCache.size();
		}
	}

	const float warmMs = std::chrono::duration<float, std::milli>(
		std::chrono::steady_clock::now() - warmStart
	).count();
	ofLogNotice() << "[aq-zoom] RAM frame cache ready in " << ofToString(warmMs, 1)
		<< " ms; cached frames=" << ramFrameCache.size();
}

void ofApp::storeRamCachedFrame(std::size_t frameIndex, std::vector<RenderParticle>&& particles) {
	if(frameIndex >= renderFrames.size() || particles.empty() || maxRamCachedFrames == 0) {
		return;
	}

	if(ramFrameCache.find(frameIndex) == ramFrameCache.end()) {
		ramFrameCacheOrder.push_back(frameIndex);
	}
	ramFrameCache[frameIndex] = std::move(particles);

	while(ramFrameCache.size() > maxRamCachedFrames && !ramFrameCacheOrder.empty()) {
		const std::size_t evictIndex = ramFrameCacheOrder.front();
		ramFrameCacheOrder.pop_front();
		if(evictIndex == currentFrameIndex && ramFrameCache.size() <= maxRamCachedFrames + 1) {
			ramFrameCacheOrder.push_back(evictIndex);
			break;
		}
		ramFrameCache.erase(evictIndex);
	}
}

const std::vector<ofApp::RenderParticle>* ofApp::findRamCachedFrame(std::size_t frameIndex) const {
	const auto found = ramFrameCache.find(frameIndex);
	return found == ramFrameCache.end() ? nullptr : &found->second;
}

std::vector<ofApp::RenderParticle> ofApp::buildRenderParticles(
	const SnapshotFrame& frame,
	const glm::vec3& rawCenter,
	float rawScale,
	const std::unordered_map<std::uint64_t, std::size_t>& slotById,
	std::size_t slotCount
) const {
	std::vector<RenderParticle> particles(slotCount);
	if(frame.particles.empty() || slotById.empty() || slotCount == 0) {
		return particles;
	}
	for(auto& particle : particles) {
		particle.typeIntensity = 5.0f;
	}

	std::size_t filledCount = 0;
	for(const auto& source : frame.particles) {
		const auto found = slotById.find(source.id);
		if(found == slotById.end() || found->second >= particles.size()) {
			continue;
		}
		const std::size_t slot = found->second;
		const glm::vec3 currentPosition = (source.position - rawCenter) * rawScale;

		particles[slot].position = currentPosition;
		particles[slot].nextPosition = currentPosition;
		particles[slot].typeIntensity = packedTypeIntensity(source);
		++filledCount;
	}

	if(filledCount == 0) {
		particles.clear();
	}
	return particles;
}

bool ofApp::loadRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedRenderFrameIndex == frameIndex && frameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadCurrentFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[aq-zoom] Failed to load cached render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadCurrentFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	if(shouldLogFrameNumber(renderFrames[frameIndex].frameNumber)) {
		ofLogNotice() << "[aq-zoom] Loaded cached render frame "
			<< renderFrames[frameIndex].frameNumber
			<< " into RAM with " << renderFrames[frameIndex].particleCount << " particles";
	}
	return true;
}

bool ofApp::loadNextRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedNextRenderFrameIndex == frameIndex && nextFrameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadNextFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[aq-zoom] Failed to load cached next render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadNextFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	return true;
}

bool ofApp::loadAheadRenderFrame(std::size_t frameIndex) {
	if(frameIndex >= renderFrames.size()) {
		return false;
	}
	if(loadedAheadRenderFrameIndex == frameIndex && aheadFrameTexture.isAllocated()) {
		return true;
	}

	if(const auto* cachedParticles = findRamCachedFrame(frameIndex)) {
		uploadAheadFrame(frameIndex, *cachedParticles);
		return true;
	}

	std::vector<RenderParticle> particles;
	if(!loadCachedFrameData(frameIndex, &particles)) {
		ofLogError() << "[aq-zoom] Failed to load cached ahead render frame "
			<< renderFrames[frameIndex].frameNumber;
		return false;
	}

	uploadAheadFrame(frameIndex, particles);
	storeRamCachedFrame(frameIndex, std::move(particles));
	return true;
}

void ofApp::uploadCurrentFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(frameTexture, frameUploadPixels, frameUploadedTexelCount, particles, true);
	updateInitialCameraFocusFromParticles(particles);
	loadedRenderFrameIndex = frameIndex;
}

void ofApp::uploadNextFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(nextFrameTexture, nextFrameUploadPixels, nextFrameUploadedTexelCount, particles, true);
	loadedNextRenderFrameIndex = frameIndex;
}

void ofApp::uploadAheadFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles) {
	populateTextureFromParticles(aheadFrameTexture, aheadFrameUploadPixels, aheadFrameUploadedTexelCount, particles, true);
	loadedAheadRenderFrameIndex = frameIndex;
}

void ofApp::populateTextureFromParticles(
	ofTexture& texture,
	ofFloatPixels& pixels,
	std::size_t& uploadedTexelCount,
	const std::vector<RenderParticle>& particles,
	bool clearUnusedTexels
) {
	bool allocatedPixels = false;
	if(pixels.getWidth() != textureWidth ||
	   pixels.getHeight() != textureHeight ||
	   pixels.getNumChannels() != 4) {
		pixels.allocate(textureWidth, textureHeight, OF_PIXELS_RGBA);
		uploadedTexelCount = 0;
		allocatedPixels = true;
	}

	float* data = pixels.getData();
	const std::size_t textureTexelCount =
		static_cast<std::size_t>(textureWidth) * static_cast<std::size_t>(textureHeight);
	const std::size_t count = std::min<std::size_t>(
		particles.size(),
		textureTexelCount
	);
	if(clearUnusedTexels && allocatedPixels) {
		std::fill(data, data + textureTexelCount * 4, 0.0f);
	} else if(clearUnusedTexels && count < uploadedTexelCount) {
		std::fill(data + count * 4, data + uploadedTexelCount * 4, 0.0f);
	}

	for(std::size_t i = 0; i < count; ++i) {
		const std::size_t offset = i * 4;
		const glm::vec3& position = particles[i].position;
		data[offset + 0] = position.x;
		data[offset + 1] = position.y;
		data[offset + 2] = position.z;
		data[offset + 3] = particles[i].typeIntensity;
	}
	uploadedTexelCount = count;

	if(!texture.isAllocated() ||
	   texture.getWidth() != textureWidth ||
	   texture.getHeight() != textureHeight) {
		texture.allocate(textureWidth, textureHeight, GL_RGBA32F);
		texture.setTextureMinMagFilter(GL_NEAREST, GL_NEAREST);
	}
	texture.loadData(pixels);
}

void ofApp::rebuildDrawMesh(std::size_t particleCount) {
	drawMesh.clear();
	drawMesh.setMode(OF_PRIMITIVE_POINTS);
	drawMesh.getVertices().resize(particleCount, glm::vec3(0.0f));
}

void ofApp::syncFrameState() {
	if(renderFrames.empty() || currentFrameIndex >= renderFrames.size()) {
		currentParticleCount = 0;
		currentDisplayParticleCount = 0;
		return;
	}

	if(currentFrameIndex == loadedNextRenderFrameIndex && nextFrameTexture.isAllocated()) {
		std::swap(frameTexture, nextFrameTexture);
		loadedRenderFrameIndex = currentFrameIndex;
		loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	}

	if(!loadRenderFrame(currentFrameIndex)) {
		currentParticleCount = 0;
		currentDisplayParticleCount = 0;
		return;
	}

	const std::size_t nextFrameIndex = (currentFrameIndex + 1) % renderFrames.size();
	if(renderFrames.size() > 1) {
		if(nextFrameIndex == loadedAheadRenderFrameIndex && aheadFrameTexture.isAllocated()) {
			std::swap(nextFrameTexture, aheadFrameTexture);
			loadedNextRenderFrameIndex = nextFrameIndex;
			loadedAheadRenderFrameIndex = std::numeric_limits<std::size_t>::max();
		}
		if(loadedNextRenderFrameIndex != nextFrameIndex || !nextFrameTexture.isAllocated()) {
			if(!loadNextRenderFrame(nextFrameIndex)) {
				nextFrameTexture.clear();
				loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
			}
		}
		beginNextFramePrefetch((nextFrameIndex + 1) % renderFrames.size());
		if(!playing && renderFrames.size() > 2) {
			loadAheadRenderFrame((nextFrameIndex + 1) % renderFrames.size());
		}
	} else {
		nextFrameTexture = frameTexture;
		loadedNextRenderFrameIndex = currentFrameIndex;
	}

	currentParticleCount = renderFrames[currentFrameIndex].particleCount;
	const std::size_t renderParticleCount = getInterpolatedParticleCount();
	displayStride = computeDisplayStride(renderParticleCount);
	const std::size_t nextDisplayParticleCount = getDisplayParticleCount(renderParticleCount);
	if(nextDisplayParticleCount != currentDisplayParticleCount) {
		currentDisplayParticleCount = nextDisplayParticleCount;
		rebuildDrawMesh(currentDisplayParticleCount);
	}
}

void ofApp::prepareAheadFrameTexture() {
	if(!playing || renderFrames.size() <= 2 || currentFrameIndex >= renderFrames.size()) {
		return;
	}

	const std::size_t nextFrameIndex = (currentFrameIndex + 1) % renderFrames.size();
	const std::size_t aheadFrameIndex = (nextFrameIndex + 1) % renderFrames.size();
	if(loadedAheadRenderFrameIndex == aheadFrameIndex && aheadFrameTexture.isAllocated()) {
		return;
	}

	beginNextFramePrefetch(aheadFrameIndex);

	// Move the expensive GPU upload away from the frame boundary. At the boundary
	// syncFrameState can then promote next/ahead textures with swaps only.
	if(currentFrameBlend < 0.32f || currentFrameBlend > 0.72f) {
		return;
	}

	const auto* cachedParticles = findRamCachedFrame(aheadFrameIndex);
	if(cachedParticles == nullptr) {
		return;
	}
	uploadAheadFrame(aheadFrameIndex, *cachedParticles);
	beginNextFramePrefetch((aheadFrameIndex + 1) % renderFrames.size());
}

void ofApp::beginNextFramePrefetch(std::size_t frameIndex) {
	if(renderFrames.size() <= 1 || frameIndex >= renderFrames.size()) {
		return;
	}
	if(nextFramePrefetchFuture.valid()) {
		const auto status = nextFramePrefetchFuture.wait_for(std::chrono::milliseconds(0));
		if(status != std::future_status::ready) {
			return;
		}
		PrefetchedFrameData prefetched = nextFramePrefetchFuture.get();
		if(prefetched.frameIndex != std::numeric_limits<std::size_t>::max() && !prefetched.particles.empty()) {
			storeRamCachedFrame(prefetched.frameIndex, std::move(prefetched.particles));
		}
	}

	std::size_t targetFrameIndex = frameIndex;
	const std::size_t lookAheadCount = std::min<std::size_t>(maxRamCachedFrames, renderFrames.size());
	bool foundTarget = false;
	for(std::size_t offset = 0; offset < lookAheadCount; ++offset) {
		const std::size_t candidate = (frameIndex + offset) % renderFrames.size();
		if(candidate == loadedRenderFrameIndex || findRamCachedFrame(candidate) != nullptr) {
			continue;
		}
		targetFrameIndex = candidate;
		foundTarget = true;
		break;
	}
	if(!foundTarget) {
		return;
	}

	nextFramePrefetchFuture = std::async(std::launch::async, [this, targetFrameIndex]() {
		PrefetchedFrameData prefetched;
		prefetched.frameIndex = targetFrameIndex;
		if(!loadCachedFrameData(targetFrameIndex, &prefetched.particles)) {
			prefetched.frameIndex = std::numeric_limits<std::size_t>::max();
			prefetched.particles.clear();
		}
		return prefetched;
	});
}

void ofApp::finishNextFramePrefetch() {
	if(!nextFramePrefetchFuture.valid()) {
		return;
	}
	const auto status = nextFramePrefetchFuture.wait_for(std::chrono::milliseconds(0));
	if(status != std::future_status::ready) {
		return;
	}

	PrefetchedFrameData prefetched = nextFramePrefetchFuture.get();
	if(prefetched.frameIndex == std::numeric_limits<std::size_t>::max() || prefetched.particles.empty()) {
		return;
	}
	storeRamCachedFrame(prefetched.frameIndex, std::move(prefetched.particles));
}

void ofApp::waitForPendingPrefetch() {
	if(!nextFramePrefetchFuture.valid()) {
		return;
	}
	nextFramePrefetchFuture.wait();
	try {
		(void)nextFramePrefetchFuture.get();
	} catch(...) {
	}
}

void ofApp::updatePlaybackPosition(double deltaSeconds) {
	if(renderFrames.empty()) {
		return;
	}
	deltaSeconds = std::min(deltaSeconds, 1.0 / 30.0);
	const double interval = 1.0 / static_cast<double>(std::max(0.001f, playbackFpsParam.get()));
	const double frameStep = static_cast<double>(std::max(1, frameStepParam.get()));
	playbackFramePosition += (deltaSeconds / interval) * frameStep;
	const double frameCount = static_cast<double>(renderFrames.size());
	playbackFramePosition = std::fmod(playbackFramePosition, frameCount);
	if(playbackFramePosition < 0.0) {
		playbackFramePosition += frameCount;
	}
}

void ofApp::applyPlaybackPosition(bool reloadRenderFrame) {
	if(renderFrames.empty()) {
		return;
	}

	std::size_t nextFrameIndex = currentFrameIndex;
	float nextBlend = currentFrameBlend;
	frameStateForPosition(playbackFramePosition, nextFrameIndex, nextBlend);
	const bool frameChanged = nextFrameIndex != currentFrameIndex;
	currentFrameIndex = nextFrameIndex;
	currentFrameBlend = nextBlend;
	if(reloadRenderFrame && (frameChanged || !frameTexture.isAllocated())) {
		syncFrameState();
	}
}

void ofApp::frameStateForPosition(double playbackPosition, std::size_t& frameIndex, float& frameBlend) const {
	frameIndex = 0;
	frameBlend = 0.0f;
	if(renderFrames.empty()) {
		return;
	}

	const double frameCount = static_cast<double>(renderFrames.size());
	double wrapped = std::fmod(playbackPosition, frameCount);
	if(wrapped < 0.0) {
		wrapped += frameCount;
	}
	frameIndex = static_cast<std::size_t>(std::floor(wrapped)) % renderFrames.size();
	frameBlend = static_cast<float>(wrapped - std::floor(wrapped));
}

std::size_t ofApp::getInterpolatedParticleCount() const {
	if(renderFrames.empty() || currentFrameIndex >= renderFrames.size()) {
		return 0;
	}
	return currentParticleCount;
}

std::size_t ofApp::getDisplayParticleCount(std::size_t particleCount) const {
	if(particleCount == 0) {
		return 0;
	}
	return (particleCount + displayStride - 1) / displayStride;
}

bool ofApp::particleTypeEnabled(int type) const {
	switch(type) {
		case 0: return showGasParam;
		case 1: return showDarkMatterParam;
		case 2: return showDiskParam;
		case 3: return showBulgeParam;
		case 4: return showStarsParam;
		default: return true;
	}
}

float ofApp::packedTypeIntensity(const SnapshotParticle& particle) const {
	return packedTypeIntensityValue(particle.type, particle.intensity);
}

void ofApp::syncDisplaySettings() {
	const std::size_t renderParticleCount = getInterpolatedParticleCount();
	const std::size_t desiredStride = renderParticleCount == 0
		? 1
		: computeDisplayStride(renderParticleCount);
	if(maxDrawCountParam.get() == previousMaxDrawCount &&
	   fullResolutionParam.get() == previousFullResolution &&
	   autoLodParam.get() == previousAutoLod &&
	   desiredStride == displayStride) {
		return;
	}
	previousMaxDrawCount = maxDrawCountParam;
	previousFullResolution = fullResolutionParam;
	previousAutoLod = autoLodParam;

	displayStride = desiredStride;
	const std::size_t nextDisplayParticleCount = getDisplayParticleCount(renderParticleCount);
	if(nextDisplayParticleCount != currentDisplayParticleCount) {
		currentDisplayParticleCount = nextDisplayParticleCount;
		rebuildDrawMesh(currentDisplayParticleCount);
	}
}

std::size_t ofApp::computeDisplayStride(std::size_t renderParticleCount) const {
	if(renderParticleCount == 0) {
		return 1;
	}

	const std::size_t drawCap = static_cast<std::size_t>(std::max(1, maxDrawCountParam.get()));
	if(!fullResolutionParam) {
		return std::max<std::size_t>(1, (renderParticleCount + drawCap - 1) / drawCap);
	}

	if(!autoLodParam) {
		return 1;
	}

	const float sceneScale = std::max(sceneScaleParam.get(), 0.001f);
	const float viewScale = std::max(coreScaleParam.get(), 0.001f);
	const float normalizedRadius = initialCameraFocusValid
		? initialCameraRadius * (displayNormalizationReady ? displayNormalizationScale : 1.0f)
		: 0.75f;
	const float sceneRadius = std::max(normalizedRadius * sceneScale * viewScale, 0.08f);
	const float distance = std::max(cam.getDistance(), 0.001f);
	const float nearDistance = std::max(sceneRadius * 1.45f, kCameraInitialMinDistance);
	const float farDistance = std::max(sceneRadius * 3.0f, nearDistance + 0.001f);
	float lod = ofClamp((distance - nearDistance) / (farDistance - nearDistance), 0.0f, 1.0f);
	lod = lod * lod * (3.0f - 2.0f * lod);
	if(lod <= 0.001f) {
		return 1;
	}

	const float farDrawCount = static_cast<float>(std::min<std::size_t>(renderParticleCount, drawCap));
	const float targetDrawCount = ofLerp(static_cast<float>(renderParticleCount), farDrawCount, lod);
	return std::max<std::size_t>(
		1,
		(renderParticleCount + static_cast<std::size_t>(std::max(1.0f, targetDrawCount)) - 1) /
		static_cast<std::size_t>(std::max(1.0f, targetDrawCount))
	);
}

void ofApp::updateInitialCameraFocusFromParticles(const std::vector<RenderParticle>& particles) {
	if(initialCameraFocusValid || particles.empty()) {
		return;
	}

	auto computeFocus = [&particles](bool excludeDarkMatter, glm::vec3& focus, float& radius) {
		glm::dvec3 sum(0.0);
		std::size_t count = 0;
		for(const auto& particle : particles) {
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(excludeDarkMatter && type == 1) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			sum += glm::dvec3(position);
			++count;
		}

		if(count == 0) {
			return false;
		}

		focus = glm::vec3(sum / static_cast<double>(count));
		double sumRadiusSquared = 0.0;
		for(const auto& particle : particles) {
			const int type = static_cast<int>(std::floor(particle.typeIntensity + 0.0001f));
			if(excludeDarkMatter && type == 1) {
				continue;
			}
			const glm::vec3& position = particle.position;
			if(!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
				continue;
			}
			const glm::vec3 delta = position - focus;
			sumRadiusSquared += static_cast<double>(glm::dot(delta, delta));
		}

		radius = std::sqrt(static_cast<float>(sumRadiusSquared / static_cast<double>(count)));
		return radius > std::numeric_limits<float>::epsilon();
	};

	glm::vec3 focus(0.0f);
	float radius = 1.0f;
	if(!computeFocus(true, focus, radius) && !computeFocus(false, focus, radius)) {
		return;
	}

	initialCameraFocus = focus;
	initialCameraRadius = std::max(radius, 0.05f);
	initialCameraFocusValid = true;
	ofLogNotice() << "[aq-zoom] Initial camera focus "
		<< initialCameraFocus.x << ", "
		<< initialCameraFocus.y << ", "
		<< initialCameraFocus.z
		<< " radius=" << initialCameraRadius;
}

glm::vec3 ofApp::transformCachedPositionForDisplay(const glm::vec3& position) const {
	if(!displayNormalizationReady) {
		return position;
	}
	return (position - displayNormalizationCenter) * displayNormalizationScale;
}

void ofApp::resetCameraView() {
	cam.setAutoDistance(false);
	const float sceneScale = std::max(sceneScaleParam.get(), 0.001f);
	const float viewScale = std::max(coreScaleParam.get(), 0.001f);
	const glm::vec3 target = initialCameraFocusValid
		? transformCachedPositionForDisplay(initialCameraFocus) * sceneScale * viewScale
		: glm::vec3(0.0f);
	const float normalizedRadius = initialCameraRadius *
		(displayNormalizationReady ? displayNormalizationScale : 1.0f);
	const float distance = initialCameraFocusValid
		? ofClamp(
			normalizedRadius * sceneScale * viewScale * kCameraFocusDistanceFactor,
			kCameraInitialMinDistance,
			kCameraInitialMaxDistance
		)
		: kCameraInitialDistance;
	const glm::vec3 viewDirection = glm::normalize(glm::vec3(0.42f, -0.18f, 1.0f));

	cam.setTarget(target);
	cam.setPosition(target + viewDirection * distance);
	cam.lookAt(target, glm::vec3(0.0f, 1.0f, 0.0f));
	cam.setDistance(distance);
	cam.setNearClip(kCameraNearClip);
	cam.setFarClip(kCameraFarClip);
	cam.enableMouseInput();
}

void ofApp::constrainCameraDistance() {
	const float distance = cam.getDistance();
	if(distance < kCameraDollyDistance) {
		const glm::vec3 zAxis = cam.getZAxis();
		if(glm::length(zAxis) > std::numeric_limits<float>::epsilon()) {
			const glm::vec3 forward = -glm::normalize(zAxis);
			const glm::vec3 targetPosition = cam.getTarget().getPosition();
			cam.setTarget(targetPosition + forward * (kCameraDollyDistance - distance));
		}
		cam.setDistance(kCameraDollyDistance);
	} else if(distance > kCameraMaxDistance) {
		cam.setDistance(kCameraMaxDistance);
	}
	cam.setNearClip(kCameraNearClip);
	cam.setFarClip(kCameraFarClip);
}

void ofApp::drawPointCloud() {
	if(currentDisplayParticleCount == 0 || !frameTexture.isAllocated()) {
		return;
	}

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	if(depthParticlesParam) {
		glEnable(GL_DEPTH_TEST);
	} else {
		glDisable(GL_DEPTH_TEST);
	}
	glDepthMask(GL_FALSE);

	const float strideForDensity = densityCompensationParam
		? static_cast<float>(std::max<std::size_t>(1, displayStride))
		: 1.0f;
	const float targetDrawCount = static_cast<float>(std::max(1, maxDrawCountParam.get()));
	const float actualDrawCount = static_cast<float>(std::max<std::size_t>(1, currentDisplayParticleCount));
	const float highLoadPointScale = ofClamp(std::sqrt(targetDrawCount / actualDrawCount), 0.62f, 1.0f);
	const float pointDensityScale =
		ofClamp(std::pow(strideForDensity, 0.14f), 1.0f, 1.55f) * highLoadPointScale;
	const float densityGain = ofClamp(std::pow(strideForDensity, 0.70f), 1.0f, 4.5f);

	pointShader.begin();
	pointShader.setUniformTexture("positionTexture", frameTexture, 0);
	pointShader.setUniformTexture(
		"nextPositionTexture",
		nextFrameTexture.isAllocated() ? nextFrameTexture : frameTexture,
		1
	);
	pointShader.setUniform1i("textureWidth", textureWidth);
	pointShader.setUniform1i("displayStride", static_cast<int>(displayStride));
	pointShader.setUniform1f("pointSize", pointSizeParam);
	pointShader.setUniform1f("pointDensityScale", pointDensityScale);
	pointShader.setUniform1f("sceneScale", sceneScaleParam);
	pointShader.setUniform1f("viewScale", coreScaleParam);
	pointShader.setUniform3f(
		"displayNormalizationCenter",
		displayNormalizationReady ? displayNormalizationCenter : glm::vec3(0.0f)
	);
	pointShader.setUniform1f(
		"displayNormalizationScale",
		displayNormalizationReady ? displayNormalizationScale : 1.0f
	);
	pointShader.setUniform1f("frameBlend", currentFrameBlend);
	pointShader.setUniform1f("brightness", brightnessParam);
	pointShader.setUniform1f("densityGain", densityGain);
	pointShader.setUniform1f("gradientMix", gradientParam);
	pointShader.setUniform1f("typeColorMix", typeColorParam);
	pointShader.setUniform1i("fastFullRes", fullResolutionParam && performanceModeParam ? 1 : 0);
	pointShader.setUniform1i("performanceMode", performanceModeParam ? 1 : 0);
	pointShader.setUniform1i("temporalSmooth", temporalSmoothParam ? 1 : 0);
	pointShader.setUniform1i("showGas", showGasParam ? 1 : 0);
	pointShader.setUniform1i("showDarkMatter", showDarkMatterParam ? 1 : 0);
	pointShader.setUniform1i("showDisk", showDiskParam ? 1 : 0);
	pointShader.setUniform1i("showBulge", showBulgeParam ? 1 : 0);
	pointShader.setUniform1i("showStars", showStarsParam ? 1 : 0);
	pointShader.setUniformMatrix4f("modelViewMatrix", ofGetCurrentMatrix(OF_MATRIX_MODELVIEW));
	pointShader.setUniformMatrix4f("projectionMatrix", ofGetCurrentMatrix(OF_MATRIX_PROJECTION));

	drawMesh.draw();

	pointShader.end();
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthFunc(GL_LESS);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
}

void ofApp::drawBounds() const {
	ofPushStyle();
	ofNoFill();
	ofSetColor(255, 44);
	const float sceneScale = sceneScaleParam.get();
	const glm::vec3 worldCenter = 0.5f * (worldMin + worldMax) * sceneScale;
	const glm::vec3 worldSize = (worldMax - worldMin) * sceneScale;
	ofDrawBox(worldCenter, worldSize.x, worldSize.y, worldSize.z);
	ofPopStyle();
}

void ofApp::drawHud() const {
	ofDisableDepthTest();
	ofSetColor(255);

	std::ostringstream text;
	text << std::fixed << std::setprecision(3);
	if(preloadReady && !renderFrames.empty() && currentFrameIndex < renderFrames.size()) {
		const auto& frame = renderFrames[currentFrameIndex];
		text << "frame " << frame.frameNumber
			<< "  cached " << (currentFrameIndex + 1) << " / " << renderFrames.size()
			<< "  t=" << frame.time
			<< "  draw=" << currentDisplayParticleCount
			<< " / cached=" << currentParticleCount
			<< '\n';
		text << (playing ? "playing" : "paused")
			<< "  blend=" << currentFrameBlend
			<< "  stride=" << displayStride
			<< "  mode=" << (performanceModeParam ? "fast" : "quality")
			<< "  fullRes=" << (fullResolutionParam ? "on" : "off")
			<< "  ram=" << ramFrameCache.size() << "/" << maxRamCachedFrames
			<< "  " << statusMessage;
	} else {
		text << statusMessage;
	}
	ofDrawBitmapStringHighlight(text.str(), 18.0f, ofGetHeight() - 44.0f);
	ofEnableDepthTest();
}
