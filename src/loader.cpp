#include "loader.h"

#include <H5Cpp.h>
#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace H5;

namespace {
constexpr int kParticleTypeCount = 6;

bool hdf5LinkExists(hid_t loc, const std::string& name) {
	htri_t exists = 0;
	H5E_BEGIN_TRY {
		exists = H5Lexists(loc, name.c_str(), H5P_DEFAULT);
	} H5E_END_TRY;
	return exists > 0;
}

bool hdf5AttrExists(hid_t loc, const char* name) {
	htri_t exists = 0;
	H5E_BEGIN_TRY {
		exists = H5Aexists(loc, name);
	} H5E_END_TRY;
	return exists > 0;
}

double readDoubleAttributeOr(const Group& group, const char* name, double fallback) {
	if(!hdf5AttrExists(group.getId(), name)) {
		return fallback;
	}

	try {
		double value = fallback;
		Attribute attr = group.openAttribute(name);
		attr.read(PredType::NATIVE_DOUBLE, &value);
		return value;
	} catch(const Exception&) {
		return fallback;
	}
}

std::vector<float> readFloatVectorDataset(const Group& group, const std::string& name, std::size_t expectedCount) {
	if(!hdf5LinkExists(group.getId(), name)) {
		return {};
	}

	DataSet dataSet = group.openDataSet(name);
	DataSpace space = dataSet.getSpace();
	if(space.getSimpleExtentNdims() != 1) {
		return {};
	}

	hsize_t dims[1] = {0};
	space.getSimpleExtentDims(dims, nullptr);
	if(static_cast<std::size_t>(dims[0]) != expectedCount) {
		return {};
	}

	std::vector<float> values(expectedCount, 0.0f);
	dataSet.read(values.data(), PredType::NATIVE_FLOAT);
	return values;
}

std::vector<std::uint64_t> readParticleIds(const Group& group, std::size_t expectedCount) {
	std::vector<std::uint64_t> ids(expectedCount, 0);
	std::iota(ids.begin(), ids.end(), 0);
	if(!hdf5LinkExists(group.getId(), "ParticleIDs")) {
		return ids;
	}

	DataSet dataSet = group.openDataSet("ParticleIDs");
	DataSpace space = dataSet.getSpace();
	if(space.getSimpleExtentNdims() != 1) {
		return ids;
	}

	hsize_t dims[1] = {0};
	space.getSimpleExtentDims(dims, nullptr);
	if(static_cast<std::size_t>(dims[0]) != expectedCount) {
		return ids;
	}

	dataSet.read(ids.data(), PredType::NATIVE_UINT64);
	return ids;
}

void appendParticleType(
	H5File& file,
	int type,
	SnapshotFrame& frame
) {
	const std::array<std::string, 2> groupCandidates = {
		"/PartType" + ofToString(type),
		"/ParticleType" + ofToString(type)
	};

	std::string groupPath;
	for(const auto& candidate : groupCandidates) {
		if(hdf5LinkExists(file.getId(), candidate) &&
		   hdf5LinkExists(file.getId(), candidate + "/Coordinates")) {
			groupPath = candidate;
			break;
		}
	}
	if(groupPath.empty()) {
		return;
	}

	Group group = file.openGroup(groupPath);
	DataSet coordinateSet = group.openDataSet("Coordinates");
	DataSpace coordinateSpace = coordinateSet.getSpace();
	if(coordinateSpace.getSimpleExtentNdims() != 2) {
		throw std::runtime_error(groupPath + "/Coordinates must have shape (N, 3)");
	}

	hsize_t dims[2] = {0, 0};
	coordinateSpace.getSimpleExtentDims(dims, nullptr);
	if(dims[1] != 3) {
		throw std::runtime_error(groupPath + "/Coordinates must have shape (N, 3)");
	}

	const std::size_t particleCount = static_cast<std::size_t>(dims[0]);
	if(particleCount == 0) {
		return;
	}

	std::vector<float> rawCoordinates(particleCount * 3, 0.0f);
	coordinateSet.read(rawCoordinates.data(), PredType::NATIVE_FLOAT);
	std::vector<std::uint64_t> ids = readParticleIds(group, particleCount);

	std::vector<float> sfr;
	float maxSfr = 0.0f;
	if(type == 0) {
		sfr = readFloatVectorDataset(group, "StarFormationRate", particleCount);
		for(float value : sfr) {
			if(std::isfinite(value)) {
				maxSfr = std::max(maxSfr, value);
			}
		}
	}

	std::vector<SnapshotParticle> particles;
	particles.reserve(particleCount);
	for(std::size_t i = 0; i < particleCount; ++i) {
		SnapshotParticle particle;
		particle.position = glm::vec3(
			rawCoordinates[i * 3 + 0],
			rawCoordinates[i * 3 + 1],
			rawCoordinates[i * 3 + 2]
		);
		particle.id = ids[i];
		particle.type = type;
		if(type == 0 && i < sfr.size() && maxSfr > std::numeric_limits<float>::epsilon()) {
			particle.intensity = ofClamp(
				std::log1p(std::max(0.0f, sfr[i])) / std::log1p(maxSfr),
				0.0f,
				1.0f
			);
		}

		if(frame.boundsValid) {
			frame.rawMin = glm::min(frame.rawMin, particle.position);
			frame.rawMax = glm::max(frame.rawMax, particle.position);
		} else {
			frame.rawMin = particle.position;
			frame.rawMax = particle.position;
			frame.boundsValid = true;
		}
		particles.push_back(particle);
	}

	std::sort(particles.begin(), particles.end(), [](const SnapshotParticle& a, const SnapshotParticle& b) {
		return a.id < b.id;
	});

	frame.typeCounts[static_cast<std::size_t>(type)] += particles.size();
	frame.particles.insert(frame.particles.end(), particles.begin(), particles.end());
}
}

int GadgetSnapshotLoader::parseFrameNumber(const std::string& fileName) const {
	const std::string suffix = ".hdf5";
	if(fileName.size() <= suffix.size() || fileName.rfind(suffix) != fileName.size() - suffix.size()) {
		return -1;
	}

	const std::string stem = fileName.substr(0, fileName.size() - suffix.size());
	if(stem.rfind("snapshot_", 0) != 0) {
		return -1;
	}

	std::string numberText = stem.substr(std::string("snapshot_").size());
	const std::size_t partSeparator = numberText.find('.');
	if(partSeparator != std::string::npos) {
		numberText = numberText.substr(0, partSeparator);
	}
	if(numberText.empty()) {
		return -1;
	}
	for(char c : numberText) {
		if(!std::isdigit(static_cast<unsigned char>(c))) {
			return -1;
		}
	}

	return std::stoi(numberText);
}

std::vector<SnapshotFrameInfo> GadgetSnapshotLoader::discoverFrames(const std::string& directoryPath) const {
	namespace fs = std::filesystem;

	fs::path root(directoryPath);
	std::error_code ec;
	if(fs::is_directory(root / "output", ec)) {
		root /= "output";
	}

	std::vector<SnapshotFrameInfo> frames;
	if(!fs::is_directory(root, ec)) {
		ofLogError() << "[aq-zoom] Snapshot directory not found: " << root.string();
		return frames;
	}

	ofLogNotice() << "[aq-zoom] Scanning snapshots under " << root.string();
	for(fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
		it != end;
		it.increment(ec)) {
		if(ec) {
			ec.clear();
			continue;
		}
		if(!it->is_regular_file(ec)) {
			continue;
		}

		const std::string fileName = it->path().filename().string();
		const int frameNumber = parseFrameNumber(fileName);
		if(frameNumber < 0) {
			continue;
		}

		SnapshotFrameInfo info;
		info.frameNumber = frameNumber;
		info.filePath = it->path().string();
		info.partPaths.push_back(info.filePath);
		frames.push_back(std::move(info));
		if(frames.size() % 250 == 0) {
			ofLogNotice() << "[aq-zoom] Discovered " << frames.size()
				<< " snapshot files so far; latest frame " << frameNumber;
		}
	}

	std::sort(frames.begin(), frames.end(), [](const SnapshotFrameInfo& a, const SnapshotFrameInfo& b) {
		if(a.frameNumber != b.frameNumber) {
			return a.frameNumber < b.frameNumber;
		}
		return a.filePath < b.filePath;
	});

	std::size_t splitPartCount = 0;
	if(!frames.empty()) {
		std::vector<SnapshotFrameInfo> uniqueFrames;
		uniqueFrames.reserve(frames.size());
		for(const auto& frame : frames) {
			if(uniqueFrames.empty() || uniqueFrames.back().frameNumber != frame.frameNumber) {
				uniqueFrames.push_back(frame);
				continue;
			}
			++splitPartCount;
			uniqueFrames.back().partPaths.push_back(frame.filePath);
		}
		frames = std::move(uniqueFrames);
	}

	for(auto& frame : frames) {
		std::sort(frame.partPaths.begin(), frame.partPaths.end());
		if(!frame.partPaths.empty()) {
			frame.filePath = frame.partPaths.front();
		}
	}

	if(frames.empty()) {
		ofLogWarning() << "[aq-zoom] No snapshot_*.hdf5 files found under " << root.string();
	} else {
		ofLogNotice() << "[aq-zoom] Snapshot scan complete: " << frames.size()
			<< " frames, frame range " << frames.front().frameNumber
			<< "..." << frames.back().frameNumber;
		if(splitPartCount > 0) {
			ofLogNotice() << "[aq-zoom] Grouped " << (splitPartCount + frames.size())
				<< " split snapshot files into " << frames.size() << " frames";
		}
	}

	return frames;
}

SnapshotFrame GadgetSnapshotLoader::loadFrame(const SnapshotFrameInfo& frameInfo) const {
	static const bool hdf5DontPrintInitialized = []() {
		H5::Exception::dontPrint();
		return true;
	}();
	(void)hdf5DontPrintInitialized;

	try {
		SnapshotFrame frame;
		frame.frameNumber = frameInfo.frameNumber;
		frame.filePath = frameInfo.filePath;

		std::vector<std::string> paths = frameInfo.partPaths;
		if(paths.empty()) {
			paths.push_back(frameInfo.filePath);
		}
		for(const auto& path : paths) {
			H5File file(path, H5F_ACC_RDONLY);
			if(frame.time == 0.0 && hdf5LinkExists(file.getId(), "/Header")) {
				Group header = file.openGroup("/Header");
				frame.time = readDoubleAttributeOr(header, "Time", 0.0);
			}

			for(int type = 0; type < kParticleTypeCount; ++type) {
				try {
					appendParticleType(file, type, frame);
				} catch(const Exception& e) {
					ofLogWarning() << "Failed to load " << typeName(type) << " from " << path
						<< ": " << e.getDetailMsg();
				} catch(const std::exception& e) {
					ofLogWarning() << "Failed to load " << typeName(type) << " from " << path
						<< ": " << e.what();
				}
			}
		}

		std::sort(frame.particles.begin(), frame.particles.end(), [](const SnapshotParticle& a, const SnapshotParticle& b) {
			return a.id < b.id;
		});

		if(frame.particles.empty()) {
			throw std::runtime_error("No readable PartType*/Coordinates datasets in " + frameInfo.filePath);
		}

		return frame;
	} catch(const Exception& e) {
		throw std::runtime_error("HDF5 error opening/reading " + frameInfo.filePath + ": " + e.getDetailMsg());
	}
}

std::string GadgetSnapshotLoader::typeName(int type) {
	switch(type) {
		case 0: return "gas";
		case 1: return "dark matter";
		case 2: return "disk stars";
		case 3: return "bulge stars";
		case 4: return "new stars";
		case 5: return "black holes";
		default: return "type " + ofToString(type);
	}
}
