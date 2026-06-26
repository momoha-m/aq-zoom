#pragma once

#include "ofMain.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct SnapshotParticle {
	glm::vec3 position = glm::vec3(0.0f);
	std::uint64_t id = 0;
	int type = 0;
	float intensity = 0.0f;
};

struct SnapshotFrameInfo {
	int frameNumber = 0;
	std::string filePath;
	std::vector<std::string> partPaths;
};

struct SnapshotFrame {
	int frameNumber = 0;
	double time = 0.0;
	std::string filePath;
	std::vector<SnapshotParticle> particles;
	std::array<std::size_t, 6> typeCounts{};
	glm::vec3 rawMin = glm::vec3(0.0f);
	glm::vec3 rawMax = glm::vec3(0.0f);
	bool boundsValid = false;
};

class GadgetSnapshotLoader {
public:
	std::vector<SnapshotFrameInfo> discoverFrames(const std::string& directoryPath) const;
	SnapshotFrame loadFrame(const SnapshotFrameInfo& frameInfo) const;
	static std::string typeName(int type);

private:
	int parseFrameNumber(const std::string& fileName) const;
};
