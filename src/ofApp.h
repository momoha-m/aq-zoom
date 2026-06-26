#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "loader.h"

#include <array>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void draw() override;
	void exit() override;

	void keyPressed(int key) override;
	void windowResized(int w, int h) override;

private:
	struct CachedFrameInfo {
		std::size_t sourceIndex = 0;
		int frameNumber = 0;
		double time = 0.0;
		std::size_t particleCount = 0;
		std::array<std::size_t, 6> typeCounts{};
	};

	struct RenderParticle {
		glm::vec3 position = glm::vec3(0.0f);
		glm::vec3 nextPosition = glm::vec3(0.0f);
		float typeIntensity = 0.0f;
	};

	struct PrefetchedFrameData {
		std::size_t frameIndex = std::numeric_limits<std::size_t>::max();
		std::vector<RenderParticle> particles;
	};

	void setupGui();
	void setupPointShader();
	std::string resolveDataDirectoryPath() const;

	bool preloadAllFrameData();
	std::filesystem::path buildCacheDirectory() const;
	bool loadPreprocessedCache(const std::filesystem::path& cacheDir);
	bool writePreprocessedCache(const std::filesystem::path& cacheDir) const;
	bool loadCachedFrameData(std::size_t frameIndex, std::vector<RenderParticle>* particles) const;
	bool writeCachedFrameData(std::size_t frameIndex, const std::vector<RenderParticle>& particles) const;
	bool computeDisplayNormalizationFromCache();
	void warmInitialRamCache(std::size_t startFrameIndex);
	void storeRamCachedFrame(std::size_t frameIndex, std::vector<RenderParticle>&& particles);
	const std::vector<RenderParticle>* findRamCachedFrame(std::size_t frameIndex) const;
	std::vector<RenderParticle> buildRenderParticles(
		const SnapshotFrame& frame,
		const glm::vec3& rawCenter,
		float rawScale,
		const std::unordered_map<std::uint64_t, std::size_t>& slotById,
		std::size_t slotCount
	) const;

	bool loadRenderFrame(std::size_t frameIndex);
	bool loadNextRenderFrame(std::size_t frameIndex);
	bool loadAheadRenderFrame(std::size_t frameIndex);
	void uploadCurrentFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void uploadNextFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void uploadAheadFrame(std::size_t frameIndex, const std::vector<RenderParticle>& particles);
	void populateTextureFromParticles(
		ofTexture& texture,
		ofFloatPixels& pixels,
		std::size_t& uploadedTexelCount,
		const std::vector<RenderParticle>& particles,
		bool clearUnusedTexels
	);
	void rebuildDrawMesh(std::size_t particleCount);
	void syncFrameState();
	void prepareAheadFrameTexture();
	void beginNextFramePrefetch(std::size_t frameIndex);
	void finishNextFramePrefetch();
	void waitForPendingPrefetch();
	void updatePlaybackPosition(double deltaSeconds);
	void applyPlaybackPosition(bool reloadRenderFrame);
	void frameStateForPosition(double playbackPosition, std::size_t& frameIndex, float& frameBlend) const;
	std::size_t getInterpolatedParticleCount() const;
	std::size_t getDisplayParticleCount(std::size_t particleCount) const;
	bool particleTypeEnabled(int type) const;
	float packedTypeIntensity(const SnapshotParticle& particle) const;
	void syncDisplaySettings();
	std::size_t computeDisplayStride(std::size_t renderParticleCount) const;
	void updateInitialCameraFocusFromParticles(const std::vector<RenderParticle>& particles);
	glm::vec3 transformCachedPositionForDisplay(const glm::vec3& position) const;
	void resetCameraView();
	void constrainCameraDistance();
	void drawPointCloud();
	void drawBounds() const;
	void drawHud() const;

	const std::string defaultDataDirectoryPath =
		"/Users/momoha/dropbox_downloads/gadget-4/DM-Zoom-Aq-C-5";
	std::string dataDirectoryPath;

	GadgetSnapshotLoader loader;
	std::vector<SnapshotFrameInfo> frameInfos;
	std::vector<CachedFrameInfo> renderFrames;
	std::filesystem::path cacheDirectory;

	ofEasyCam cam;
	ofVboMesh drawMesh;
	ofShader pointShader;
	ofTexture frameTexture;
	ofTexture nextFrameTexture;
	ofTexture aheadFrameTexture;
	ofFloatPixels frameUploadPixels;
	ofFloatPixels nextFrameUploadPixels;
	ofFloatPixels aheadFrameUploadPixels;
	std::future<PrefetchedFrameData> nextFramePrefetchFuture;
	std::optional<PrefetchedFrameData> stagedNextFrameData;
	std::unordered_map<std::size_t, std::vector<RenderParticle>> ramFrameCache;
	std::deque<std::size_t> ramFrameCacheOrder;

	ofxPanel gui;
	ofParameterGroup guiParams;
	ofParameter<float> pointSizeParam;
	ofParameter<float> sceneScaleParam;
	ofParameter<float> coreScaleParam;
	ofParameter<float> playbackFpsParam;
	ofParameter<int> frameStepParam;
	ofParameter<bool> temporalSmoothParam;
	ofParameter<float> brightnessParam;
	ofParameter<float> gradientParam;
	ofParameter<float> typeColorParam;
	ofParameter<int> maxDrawCountParam;
	ofParameter<bool> fullResolutionParam;
	ofParameter<bool> autoLodParam;
	ofParameter<bool> densityCompensationParam;
	ofParameter<bool> depthParticlesParam;
	ofParameter<bool> performanceModeParam;
	ofParameter<bool> showGasParam;
	ofParameter<bool> showDarkMatterParam;
	ofParameter<bool> showDiskParam;
	ofParameter<bool> showBulgeParam;
	ofParameter<bool> showStarsParam;
	ofParameter<bool> showBoundsParam;
	ofParameter<bool> autoRotateParam;
	ofParameter<bool> showGuiParam;

	glm::vec3 worldMin = glm::vec3(-1.35f, -1.05f, -1.05f);
	glm::vec3 worldMax = glm::vec3(1.35f, 1.05f, 1.05f);
	std::size_t currentFrameIndex = 0;
	std::size_t currentParticleCount = 0;
	std::size_t currentDisplayParticleCount = 0;
	std::size_t maxParticleCount = 0;
	std::size_t loadedRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t loadedNextRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t loadedAheadRenderFrameIndex = std::numeric_limits<std::size_t>::max();
	std::size_t frameUploadedTexelCount = 0;
	std::size_t nextFrameUploadedTexelCount = 0;
	std::size_t aheadFrameUploadedTexelCount = 0;
	std::size_t displayStride = 1;
	int textureWidth = 1;
	int textureHeight = 1;
	bool playing = false;
	bool preloadReady = false;
	double lastUpdateTime = 0.0;
	double playbackFramePosition = 0.0;
	float currentFrameBlend = 0.0f;
	float rotationDegrees = 0.0f;
	std::string statusMessage;
	int previousMaxDrawCount = 0;
	bool previousFullResolution = false;
	bool previousAutoLod = true;
	glm::vec3 initialCameraFocus = glm::vec3(0.0f);
	float initialCameraRadius = 1.0f;
	bool initialCameraFocusValid = false;
	glm::vec3 displayNormalizationCenter = glm::vec3(0.0f);
	float displayNormalizationScale = 1.0f;
	bool displayNormalizationReady = false;

	std::size_t cacheParticleLimit = 2000000;
	std::size_t maxRamCachedFrames = 24;
};
