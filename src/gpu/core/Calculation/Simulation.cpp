//=======================================================================================
// ____          ____    __    ______     __________   __      __       __        __
// \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
//  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
//   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
//    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
//     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
//      \    \  |    |   ________________________________________________________________
//       \    \ |    |  |  ______________________________________________________________|
//        \    \|    |  |  |         __          __     __     __     ______      _______
//         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
//          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
//           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
//            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/
//
//  This file is part of VirtualFluids. VirtualFluids is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
//  for more details.
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//  SPDX-FileCopyrightText: Copyright © VirtualFluids Project contributors, see AUTHORS.md in root folder
//
//! \addtogroup gpu_Calculation Calculation
//! \ingroup gpu_core core
//! \{
//! \author Martin Schoenherr
//=======================================================================================
#include "Simulation.h"

#include <memory>
#include <string>
#include <vector>

#include <helper_timer.h>

#include <cuda_helper/DeviceInfo.h>

#include <basics/MetaData/MetaData.h>
#include <basics/MetaData/YAML_MetaData.h>
#include <basics/utilities/UbFileOutputASCII.h>

#include <logger/Logger.h>

#include <parallel/Communicator.h>
#include <parallel/MPICommunicator.h>
#include <parallel/NullCommunicator.h>

#include "BoundaryConditions/AdvectionDiffusion/CellPerturbation.h"
#include "BoundaryConditions/RelaxationZone/BoundaryRelaxationZone.h"
#include "Calculation/Calculation.h"
#include "Calculation/UpdateGrid27.h"
#include "Communication/ExchangeData27.h"
#include "Cuda/CudaMemoryManager.h"
#include "Cuda/CudaStreamManager.h"
#include "DataStructureInitializer/GridProvider.h"
#include "GridScaling/GridScalingFactory.h"
#include "Kernel/Kernel.h"
#include "Kernel/KernelFactory/KernelFactory.h"
#include "Kernel/KernelFactory/KernelFactoryImp.h"
#include "Output/AnalysisData.hpp"
#include "Output/DataWriter.h"
#include "Output/DistributionDebugWriter.h"
#include "Output/EdgeNodeDebugWriter.hpp"
#include "Output/FileWriter.h"
#include "Output/InterfaceDebugWriter.hpp"
#include "Output/MeasurePointWriter.hpp"
#include "Output/MetaDataCreator.h"
#include "Output/NeighborDebugWriter.h"
#include "Output/QDebugVtkWriter.hpp"
#include "Parameter/EdgeNodeFinder.h"
#include "Parameter/Parameter.h"
#include "PostProcessor/Calc2ndMoments.h"
#include "PostProcessor/CalcMean.h"
#include "PostProcessor/CalcTurbulenceIntensity.h"
#include "PostProcessor/Cp.h"
#include "PostProcessor/DragLift.h"
#include "PostProcessor/EnstrophyAnalyzer.h"
#include "PostProcessor/ForceCalculations.h"
#include "PostProcessor/KineticEnergyAnalyzer.h"
#include "PostProcessor/MacroscopicQuantities.cuh"
#include "PostProcessor/TurbulenceIntensity.h"
#include "PreProcessor/InitLattice.h"
#include "PreProcessor/PreProcessorFactory/PreProcessorFactory.h"
#include "PreProcessor/PreProcessorFactory/PreProcessorFactoryImp.h"
#include "PreProcessor/ReaderMeasurePoints.h"
#include "Restart/RestartObject.h"
#include "TurbulenceModels/TurbulenceModelFactory.h"

namespace vf::gpu {

std::string getFileName(const std::string& fname, int step, int myID)
{
    return std::string(fname + "_Restart_" + std::to_string(myID) + "_" + std::to_string(step));
}

Simulation::Simulation(std::shared_ptr<Parameter> para, std::shared_ptr<GridBuilder> builder,
                       const BoundaryConditionFactory* bcFactory, GridScalingFactory* scalingFactory)
    : Simulation(para, std::make_shared<CudaMemoryManager>(para), builder, bcFactory, std::make_shared<TurbulenceModelFactory>(para), scalingFactory)
{
}

Simulation::Simulation(std::shared_ptr<Parameter> para, std::shared_ptr<CudaMemoryManager> memoryManager, std::shared_ptr<GridBuilder> builder,
                       const BoundaryConditionFactory* bcFactory, GridScalingFactory* scalingFactory)
    : Simulation(para, memoryManager, builder, bcFactory, std::make_shared<TurbulenceModelFactory>(para), scalingFactory)
{
}

Simulation::Simulation(std::shared_ptr<Parameter> para, std::shared_ptr<GridBuilder> builder,
                       const BoundaryConditionFactory* bcFactory, SPtr<TurbulenceModelFactory> tmFactory,
                       GridScalingFactory* scalingFactory)
    : Simulation(para, std::make_shared<CudaMemoryManager>(para), builder, bcFactory, tmFactory, scalingFactory)
{
}

Simulation::Simulation(std::shared_ptr<Parameter> para, std::shared_ptr<CudaMemoryManager> memoryManager, std::shared_ptr<GridBuilder> builder,
                       const BoundaryConditionFactory* bcFactory, SPtr<TurbulenceModelFactory> tmFactory,
                       GridScalingFactory* scalingFactory)
    : para(para), cudaMemoryManager(memoryManager),
#ifdef VF_MPI
      communicator(*vf::parallel::MPICommunicator::getInstance())
#else
      communicator(*vf::parallel::NullCommunicator::getInstance())
#endif
      ,
      kernelFactory(std::make_unique<KernelFactoryImp>()), preProcessorFactory(std::make_shared<PreProcessorFactoryImp>()),
      dataWriter(std::make_unique<FileWriter>())
{
    auto gridGenerator = GridProvider::makeGridGenerator(builder, para, cudaMemoryManager, communicator);
    init(*gridGenerator, bcFactory, tmFactory, scalingFactory);
}

Simulation::Simulation(std::shared_ptr<Parameter> para, std::shared_ptr<CudaMemoryManager> memoryManager,
                       vf::parallel::Communicator &communicator, GridProvider &gridProvider, const BoundaryConditionFactory* bcFactory, GridScalingFactory* scalingFactory)
    : para(para), cudaMemoryManager(memoryManager), communicator(communicator), kernelFactory(std::make_unique<KernelFactoryImp>()),
      preProcessorFactory(std::make_shared<PreProcessorFactoryImp>()), dataWriter(std::make_unique<FileWriter>())
{
    this->tmFactory = SPtr<TurbulenceModelFactory>( new TurbulenceModelFactory(para) );
    init(gridProvider, bcFactory, tmFactory, scalingFactory);
}

void Simulation::init(GridProvider &gridProvider, const BoundaryConditionFactory *bcFactory, SPtr<TurbulenceModelFactory> tmFactory, GridScalingFactory *scalingFactory)
{
    gridProvider.initalGridInformations();

    const int deviceId = communicator.mapCudaDevicesOnHosts(para->getDevices(), para->getMaxDev());

    vf::cuda::verifyAndSetDevice(deviceId);

    para->initLBMSimulationParameter();

    gridProvider.allocAndCopyForcing();
    gridProvider.allocAndCopyQuadricLimiters();

    gridProvider.setDimensions();
    gridProvider.setBoundingBox();

    para->setRe(para->getVelocity() * (real)1.0 / para->getViscosity());

    restart_object = std::make_shared<ASCIIRestartObject>();

    VF_LOG_TRACE("vis_ratio:       {}", para->getViscosityRatio());
    VF_LOG_TRACE("u0_ratio:        {}", para->getVelocityRatio());
    VF_LOG_TRACE("delta_rho:       {}", para->getDensityRatio());

    cudaMemoryManager->setMemsizeGPU(0, true);
    //////////////////////////////////////////////////////////////////////////
    allocNeighborsOffsetsScalesAndBoundaries(gridProvider, bcFactory);

    //! Get tagged fluid nodes with corresponding value for CollisionTemplate from interactors
    VF_LOG_INFO("Start initializing precollision interactors");
    for (SPtr<PreCollisionInteractor>& interactor : para->getInteractors()) {
        interactor->init();
        interactor->getTaggedFluidNodes(&gridProvider);
    }

    VF_LOG_INFO("Start initializing samplers");
    for (SPtr<Sampler>& sampler : para->getSamplers()) {
        sampler->init();
        sampler->getTaggedFluidNodes(&gridProvider);
    }

    //////////////////////////////////////////////////////////////////////////
    // CUDA streams
    if (para->getUseStreams()) {
        para->getStreamManager()->registerAndLaunchStream(CudaStreamIndex::SubDomainBorder);
        para->getStreamManager()->registerAndLaunchStream(CudaStreamIndex::Bulk);
        para->getStreamManager()->createCudaEvents();
    }
    //////////////////////////////////////////////////////////////////////////

    if (para->getKernelNeedsFluidNodeIndicesToRun()) {
        gridProvider.sortFluidNodeTags();
        gridProvider.allocArrays_taggedFluidNodes();
    }
    //////////////////////////////////////////////////////////////////////////
    // Kernel init
    //////////////////////////////////////////////////////////////////////////
    VF_LOG_TRACE("make Kernels");
    kernels = kernelFactory->makeKernels(para);
    for (const auto& kernel : kernels) {
        kernel->checkKernelParameters(para->getMaxLevel(), para->getVelocity(),
                                      para->getParH(para->getMaxLevel())->viscosity);
    }

    if (para->getDiffOn()) {
        VF_LOG_TRACE("make AD Kernels");
        adKernels = kernelFactory->makeAdvectionDiffusionKernels(para);
        std::vector<PreProcessorType> preProADTypes = adKernels.at(0)->getPreProcessorTypes();
        preProcessorAD = preProcessorFactory->makePreProcessor(preProADTypes, para);
    }

    //////////////////////////////////////////////////////////////////////////
    // PreProcessor init
    //////////////////////////////////////////////////////////////////////////
    VF_LOG_TRACE("make Preprocessors");
    std::vector<PreProcessorType> preProTypes = kernels.at(0)->getPreProcessorTypes();
    preProcessor = preProcessorFactory->makePreProcessor(preProTypes, para);

    //////////////////////////////////////////////////////////////////////////
    // Allocate Memory for Drag Lift Calculation
    //////////////////////////////////////////////////////////////////////////
    if (para->getCalcDragLift())
        allocDragLift(para.get(), cudaMemoryManager.get());

    //////////////////////////////////////////////////////////////////////////
    // Mean
    //////////////////////////////////////////////////////////////////////////
    if (para->getCalcMean()) {
        VF_LOG_TRACE("alloc Calculation for Mean Values");
        if (para->getDiffOn())
            allocMeanAD(para.get(), cudaMemoryManager.get());
        else
            allocMean(para.get(), cudaMemoryManager.get());
    }

    //////////////////////////////////////////////////////////////////////////
    // Turbulence Intensity
    //////////////////////////////////////////////////////////////////////////
    if (para->getCalcTurbulenceIntensity()) {
        VF_LOG_TRACE("alloc arrays for calculating Turbulence Intensity");
        allocTurbulenceIntensity(para.get(), cudaMemoryManager.get());
    }

    //////////////////////////////////////////////////////////////////////////
    // allocate memory and initialize 2nd, 3rd and higher order moments
    //////////////////////////////////////////////////////////////////////////
    if (para->getCalc2ndOrderMoments()) {
        alloc2ndMoments(para.get(), cudaMemoryManager.get());
        init2ndMoments(para.get());
    }
    if (para->getCalc3rdOrderMoments()) {
        alloc3rdMoments(para.get(), cudaMemoryManager.get());
        init3rdMoments(para.get());
    }
    if (para->getCalcHighOrderMoments()) {
        allocHigherOrderMoments(para.get(), cudaMemoryManager.get());
        initHigherOrderMoments(para.get());
    }

    //////////////////////////////////////////////////////////////////////////
    // MeasurePoints
    //////////////////////////////////////////////////////////////////////////
    if (para->getUseMeasurePoints()) {
        VF_LOG_TRACE("read measure points");
        ReaderMeasurePoints::readMeasurePoints(para.get(), cudaMemoryManager.get());
    }

    //////////////////////////////////////////////////////////////////////////
    // enSightGold
    //////////////////////////////////////////////////////////////////////////
    // excludeGridInterfaceNodesForMirror(para, 7);
    ////VF_LOG_INFO("print case file...");
    // printCaseFile(para);
    ////VF_LOG_INFO("print geo file...");
    // printGeoFile(para, true);  //true for binary
    ////VF_LOG_INFO("done.");

    VF_LOG_INFO("init lattice...");
    initLattice(para, preProcessor, preProcessorAD, cudaMemoryManager);
    VF_LOG_INFO("done");

    //////////////////////////////////////////////////////////////////////////
    // find indices of corner nodes for multiGPU communication
    //////////////////////////////////////////////////////////////////////////
    if (para->getDevices().size() > 2) {
        VF_LOG_INFO("Find indices of edge nodes for multiGPU communication");
        findEdgeNodesCommMultiGPU(*para);
    }
    //////////////////////////////////////////////////////////////////////////
    // Memory alloc for CheckPoint / Restart
    //////////////////////////////////////////////////////////////////////////
    if (para->getDoCheckPoint() || para->getDoRestart()) {
        VF_LOG_INFO("Alloc Memory for CheckPoint / Restart");
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            cudaMemoryManager->cudaAllocFsForCheckPointAndRestart(lev);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Restart
    //////////////////////////////////////////////////////////////////////////
    if (para->getDoRestart()) {
        VF_LOG_INFO("Restart...\n...get the Object...");

        const auto name = getFileName(para->getFName(), para->getTimeDoRestart(), para->getMyProcessID());
        restart_object->deserialize(name, para);

        VF_LOG_INFO("...copy Memory for Restart...");
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            //////////////////////////////////////////////////////////////////////////
            cudaMemoryManager->cudaCopyFsForRestart(lev);
            //////////////////////////////////////////////////////////////////////////
            // macroscopic values
            calculateMacroscopicQuantities(para->getParD(lev)->velocityX, para->getParD(lev)->velocityY, para->getParD(lev)->velocityZ,
                        para->getParD(lev)->rho, para->getParD(lev)->pressure, para->getParD(lev)->typeOfGridNode,
                        para->getParD(lev)->neighborX, para->getParD(lev)->neighborY,
                        para->getParD(lev)->neighborZ, para->getParD(lev)->numberOfNodes,
                        para->getParD(lev)->numberofthreads, para->getParD(lev)->distributions.f[0],
                        para->getParD(lev)->isEvenTimestep);
            getLastCudaError("Kernel calculateMacroscopicQuantities execution failed");
        }
        VF_LOG_INFO("done.");
    }

    //////////////////////////////////////////////////////////////////////////
    // Init UpdateGrid
    //////////////////////////////////////////////////////////////////////////
    this->updateGrid27 = std::make_unique<UpdateGrid27>(para, communicator, cudaMemoryManager, kernels, adKernels, bcFactory,
                                                        tmFactory, scalingFactory);

    //////////////////////////////////////////////////////////////////////////
    // Write Initialized Files
    //////////////////////////////////////////////////////////////////////////
    VF_LOG_INFO("Write initialized Files ...");
    dataWriter->writeInit(para, cudaMemoryManager);
    VF_LOG_INFO("... done.");

#if DEBUG_FS
    VF_LOG_INFO("Allocating host memory for DistributionWriter.");
    DistributionDebugWriter::allocateDistributionsOnHost(*cudaMemoryManager);
#endif

    VF_LOG_INFO("used Device Memory: {} MB", cudaMemoryManager->getMemsizeGPU() / 1000000.0);

    performanceOutput = std::make_unique<PerformanceMeasurement>(*para);
    metaData = createMetaData(*para);
    vf::basics::logPreSimulation(metaData);
}

void Simulation::addKineticEnergyAnalyzer(uint tAnalyse)
{
    this->kineticEnergyAnalyzer = std::make_unique<KineticEnergyAnalyzer>(this->para, tAnalyse);
}

void Simulation::addEnstrophyAnalyzer(uint tAnalyse)
{
    this->enstrophyAnalyzer = std::make_unique<EnstrophyAnalyzer>(this->para, tAnalyse);
}

void Simulation::setDataWriter(std::shared_ptr<DataWriter> dataWriter_)
{
    this->dataWriter = dataWriter_;
}

void Simulation::setFactories(std::unique_ptr<KernelFactory> &&kernelFactory_,
               std::unique_ptr<PreProcessorFactory> &&preProcessorFactory_)
{
    this->kernelFactory = std::move(kernelFactory_);
    this->preProcessorFactory = std::move(preProcessorFactory_);
}

void Simulation::initTimers()
{
    previousTimestepForAveraging = para->getTimeCalcMedStart();
    previousTimestepForTurbulenceIntensityCalculation = 0;
    timestepForMeasuringPoints = 0;

    para->setStepEnsight(0);

    averageTimer.start();
}

void Simulation::allocNeighborsOffsetsScalesAndBoundaries(GridProvider &gridProvider, const BoundaryConditionFactory* bcFactory)
{
    gridProvider.allocArrays_CoordNeighborGeo();
    gridProvider.allocArrays_OffsetScale();
    gridProvider.allocArrays_BoundaryValues(bcFactory); // allocArrays_BoundaryValues() has to be called after allocArrays_OffsetScale() because of initCommunicationArraysForCommAfterFinetoCoarse()
    gridProvider.allocArrays_BoundaryQs();
}

void Simulation::run()
{
    this->initTimers();

    ////////////////////////////////////////////////////////////////////////////////
    // Time loop
    ////////////////////////////////////////////////////////////////////////////////
    for(uint timestep = para->getTimestepStart(); timestep <= para->getTimestepEnd(); timestep++) {
        this->calculateTimestep(timestep);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // printDragLift(para);
    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    ////for (int lev = para->getCoarse(); lev <= para->getFine(); lev++)
    ////{
    ////    if (para->getParH(lev)->cpTop.size() > 0)
    ////    {
    ////        printCpTop(para, lev);
    ////    }
    ////}
    // for (int lev = 7; lev <= 8; lev++)
    //{
    //    printCpTop(para, lev);
    //}
    ////printCpTop(para);
    ////printCpBottom(para);
    ////printCpBottom2(para);
    ////////////////////////////////////////////////////////////////////////////////

    //  //////////////////////////////////////////////////////////////////////////
    //  //Copy Measure Values
    // for (int lev=para->getCoarse(); lev <= para->getFine(); lev++)
    //{
    //    VF_LOG_INFO("Copy MeasurePoints at level = {}", lev);
    //    para->cudaCopyMeasurePointsToHost(lev);
    //    para->copyMeasurePointsArrayToVector(lev);
    //    VF_LOG_INFO("Write MeasurePoints at level = {}", lev);
    //    for(int j = 0; j < (int)para->getParH(lev)->MP.size(); j++)
    //    {
    //        MeasurePointWriter::writeMeasurePoints(para, lev, j, 0);
    //    }
    //}
    //  //////////////////////////////////////////////////////////////////////////
    finalize();
}

void Simulation::calculateTimestep(uint timestep)
{
    this->updateGrid27->updateGrid(0, timestep);
    ////////////////////////////////////////////////////////////////////////////////
    // run Analyzers for kinetic energy and enstrophy for TGV in 3D
    // these analyzers only work on level 0
    ////////////////////////////////////////////////////////////////////////////////
    if (this->kineticEnergyAnalyzer || this->enstrophyAnalyzer) {
        updateGrid27->exchangeData(0);
    }
    if (this->kineticEnergyAnalyzer)
        this->kineticEnergyAnalyzer->run(timestep);
    if (this->enstrophyAnalyzer)
        this->enstrophyAnalyzer->run(timestep);
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    //Calc Mean
    ////////////////////////////////////////////////////////////////////////////////
    if (para->getCalcMean() && ((int)timestep >= para->getTimeCalcMedStart()) && ((int)timestep <= para->getTimeCalcMedEnd()))
    {
        for (int lev=para->getCoarse(); lev <= para->getFine(); lev++)
        {
            //calculateMean(para->getParD(lev)->vx_SP_Med,
                  //      para->getParD(lev)->vy_SP_Med,
                  //      para->getParD(lev)->vz_SP_Med,
                  //      para->getParD(lev)->rho_SP_Med,
                  //      para->getParD(lev)->press_SP_Med,
                  //      para->getParD(lev)->geoSP,
                  //      para->getParD(lev)->neighborX_SP,
                  //      para->getParD(lev)->neighborY_SP,
                  //      para->getParD(lev)->neighborZ_SP,
                  //      para->getParD(lev)->size_Mat_SP,
                  //      para->getParD(lev)->numberofthreads,
                  //      para->getParD(lev)->d0SP.f[0],
                  //      para->getParD(lev)->evenOrOdd);
            //getLastCudaError("calculateMacroscopicQuantities execution failed");
            calculateMeanCompressible(para->getParD(lev)->meanVelocityInXdirection,
                            para->getParD(lev)->meanVelocityInYdirection,
                            para->getParD(lev)->meanVelocityInZdirection,
                            para->getParD(lev)->meanDensity,
                            para->getParD(lev)->meanPressure,
                            para->getParD(lev)->typeOfGridNode,
                            para->getParD(lev)->neighborX,
                            para->getParD(lev)->neighborY,
                            para->getParD(lev)->neighborZ,
                            para->getParD(lev)->numberOfNodes,
                            para->getParD(lev)->numberofthreads,
                            para->getParD(lev)->distributions.f[0],
                            para->getParD(lev)->isEvenTimestep);
            getLastCudaError("CalcMacMedCompSP27 execution failed");
        }
    }
    if (para->getCalcTurbulenceIntensity()) {
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            CalcTurbulenceIntensityDevice(
                para->getParD(lev)->vxx,
                para->getParD(lev)->vyy,
                para->getParD(lev)->vzz,
                para->getParD(lev)->vxy,
                para->getParD(lev)->vxz,
                para->getParD(lev)->vyz,
                para->getParD(lev)->vx_mean,
                para->getParD(lev)->vy_mean,
                para->getParD(lev)->vz_mean,
                para->getParD(lev)->distributions.f[0],
                para->getParD(lev)->typeOfGridNode,
                para->getParD(lev)->neighborX,
                para->getParD(lev)->neighborY,
                para->getParD(lev)->neighborZ,
                para->getParD(lev)->numberOfNodes,
                para->getParD(lev)->isEvenTimestep,
                para->getParD(lev)->numberofthreads
            );
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // CheckPoint
    ////////////////////////////////////////////////////////////////////////////////
    if (para->getDoCheckPoint() && para->getTimeDoCheckPoint() > 0 && timestep % para->getTimeDoCheckPoint() == 0 &&
        timestep > 0 && !para->overWritingRestart(timestep)) {
        averageTimer.end();
        VF_LOG_INFO("Copy data for CheckPoint t = {}....", timestep);
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            cudaMemoryManager->cudaCopyFsForCheckPoint(lev);
        }
        VF_LOG_INFO("Write data for CheckPoint t = {}...", timestep);
        const auto name = getFileName(para->getFName(), timestep, para->getMyProcessID());
        restart_object->serialize(name, para);
        VF_LOG_INFO("done");
        averageTimer.start();
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Measure Points
    ////////////////////////////////////////////////////////////////////////////////
    // set MP-Time
    if (para->getUseMeasurePoints()) {
        if ((timestep % para->getTimestepForMeasurePoints()) == 0) {
            unsigned int valuesPerClockCycle = (unsigned int)(para->getclockCycleForMeasurePoints() / para->getTimestepForMeasurePoints());
            for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
                calculateMeasurePoints( 
                    para->getParD(lev)->velocityInXdirectionAtMeasurePoints,
                    para->getParD(lev)->velocityInYdirectionAtMeasurePoints,
                    para->getParD(lev)->velocityInZdirectionAtMeasurePoints,
                    para->getParD(lev)->densityAtMeasurePoints,
                    para->getParD(lev)->indicesOfMeasurePoints,
                    para->getParD(lev)->numberOfMeasurePoints,
                    valuesPerClockCycle,
                    timestepForMeasuringPoints,
                    para->getParD(lev)->typeOfGridNode,
                    para->getParD(lev)->neighborX,
                    para->getParD(lev)->neighborY,
                    para->getParD(lev)->neighborZ,
                    para->getParD(lev)->numberOfNodes,
                    para->getParD(lev)->distributions.f[0],
                    para->getParD(lev)->numberofthreads,
                    para->getParD(lev)->isEvenTimestep);
            }
            timestepForMeasuringPoints++;
        }
        // Copy Measure Values
        if ((timestep % (unsigned int)para->getclockCycleForMeasurePoints()) == 0) {
            for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
                cudaMemoryManager->cudaCopyMeasurePointsToHost(lev);
                para->copyMeasurePointsArrayToVector(lev);
                VF_LOG_INFO("Write MeasurePoints at level = {} and timestep = {}", lev, timestep);
                for (int j = 0; j < (int)para->getParH(lev)->MeasurePointVector.size(); j++)
                {
                    MeasurePointWriter::writeMeasurePoints(para.get(), lev, j, timestep);
                }
                // MeasurePointWriter::calcAndWriteMeanAndFluctuations(para.get(), lev, t, para->getTStartOut());
            }
            timestepForMeasuringPoints = 0;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // File IO
    ////////////////////////////////////////////////////////////////////////////////
    if (para->getTimestepOut() > 0 && timestep % para->getTimestepOut() == 0 && timestep >= para->getTimestepStartOut()) {
        averageTimer.end();
        metaData.simulation.runtimeSeconds += averageTimer.getTimeInSeconds();
        performanceOutput->log(averageTimer, timestep, communicator);

        if (para->getPrintFiles()) {
            readAndWriteFiles(timestep);
        }
        averageTimer.start();
    }
}

void Simulation::finalize()
{
    averageTimer.end();
    metaData.simulation.runtimeSeconds += averageTimer.getTimeInSeconds();
    performanceOutput->log(averageTimer, para->getTimestepEnd(), communicator);
    metaData.simulation.nups = performanceOutput->getNups();
    metaData.simulation.runtimeSeconds = performanceOutput->totalRuntimeInSeconds();
    vf::basics::logPostSimulation(metaData);

    vf::basics::writeYAML(metaData, para->getFName() + ".yaml");
}


void Simulation::readAndWriteFiles(uint timestep)
{
    VF_LOG_INFO("Write files t = {} ...", timestep);

    for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {

        updateGrid27->exchangeData(lev);

        //    if (para->getCalcMean() && ((int)t > para->getTimeCalcMedStart()) && ((int)t <= para->getTimeCalcMedEnd()))
        //    {
        //         unsigned int tdiff = t - t_prev;
        //         calculateMacrosopicMean(para->getParD(lev)->vx_SP_Med,
        //                           para->getParD(lev)->vy_SP_Med,
        //                           para->getParD(lev)->vz_SP_Med,
        //                           para->getParD(lev)->rho_SP_Med,
        //                           para->getParD(lev)->press_SP_Med,
        //                           para->getParD(lev)->geoSP,
        //                           para->getParD(lev)->neighborX_SP,
        //                           para->getParD(lev)->neighborY_SP,
        //                           para->getParD(lev)->neighborZ_SP,
        //                           tdiff,
        //                           para->getParD(lev)->size_Mat_SP,
        //                           para->getParD(lev)->numberofthreads,
        //                           para->getParD(lev)->evenOrOdd);
        //         getLastCudaError("calculateMacrosopicMean execution failed");
        //    }
        //    calculateMacroscopicQuantities(para->getParD(lev)->vx_SP,
        //                        para->getParD(lev)->vy_SP,
        //                        para->getParD(lev)->vz_SP,
        //                        para->getParD(lev)->rho,
        //                        para->getParD(lev)->pressure,
        //                        para->getParD(lev)->geoSP,
        //                        para->getParD(lev)->neighborX_SP,
        //                        para->getParD(lev)->neighborY_SP,
        //                        para->getParD(lev)->neighborZ_SP,
        //                        para->getParD(lev)->size_Mat_SP,
        //                        para->getParD(lev)->numberofthreads,
        //                        para->getParD(lev)->d0SP.f[0],
        //                        para->getParD(lev)->evenOrOdd);
        //     getLastCudaError("calculateMacroscopicQuantities execution failed");
            calculateMacroscopicQuantitiesCompressible(para->getParD(lev)->velocityX,
                            para->getParD(lev)->velocityY,
                            para->getParD(lev)->velocityZ,
                            para->getParD(lev)->rho,
                            para->getParD(lev)->pressure,
                            para->getParD(lev)->typeOfGridNode,
                            para->getParD(lev)->neighborX,
                            para->getParD(lev)->neighborY,
                            para->getParD(lev)->neighborZ,
                            para->getParD(lev)->numberOfNodes,
                            para->getParD(lev)->numberofthreads,
                            para->getParD(lev)->distributions.f[0],
                            para->getParD(lev)->isEvenTimestep);
            getLastCudaError("calculateMacroscopicQuantities execution failed");

        cudaMemoryManager->cudaCopyPrint(lev);

        // Fix BC node velocities for VTK output:
        // calculateMacroscopicQuantitiesCompressible computed wrong velocities for BC nodes
        // (from stopper neighbors). Overwrite with correct velocityBC values.
        const auto& bcH = para->getParH(lev)->velocityBC;
        if (bcH.numberOfBCnodes > 0) {
            real* vxH = para->getParH(lev)->velocityX;
            real* vyH = para->getParH(lev)->velocityY;
            real* vzH = para->getParH(lev)->velocityZ;
            for (uint i = 0; i < bcH.numberOfBCnodes; i++) {
                const int nodeIdx = bcH.k[i];
                if (nodeIdx > 0) {
                    vxH[nodeIdx] = bcH.Vx[i];
                    vyH[nodeIdx] = bcH.Vy[i];
                    vzH[nodeIdx] = bcH.Vz[i];
                }
            }
        }

        if (para->getCalcMean()) {
            cudaMemoryManager->cudaCopyMeanPrint(lev);
        }

        //////////////////////////////////////////////////////////////////////////
        std::string fname = para->getFName() + "_ID_" + StringUtil::toString<int>(para->getMyProcessID()) + "_t_" + StringUtil::toString<int>(timestep);
        if (this->kineticEnergyAnalyzer)
            this->kineticEnergyAnalyzer->writeToFile(fname);
        if (this->enstrophyAnalyzer)
            this->enstrophyAnalyzer->writeToFile(fname);
        //////////////////////////////////////////////////////////////////////////
        if (para->getDiffOn())
            cudaMemoryManager->cudaCopyConcentrationDeviceToHost(lev);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////print cp
        //if ((para->getParH(lev)->cpTop.size() > 0) && (t > para->getTStartOut()))
        //{
           // printCpTopIntermediateStep(para, t, lev);
        //}
        ////////////////////////////////////////////////////////////////////////////////
        //MeasurePointWriter::writeSpacialAverageForXZSlices(para, lev, t);
        ////////////////////////////////////////////////////////////////////////////////
        //MeasurePointWriter::writeTestAcousticXY(para, lev, t);
        //MeasurePointWriter::writeTestAcousticYZ(para, lev, t);
        //MeasurePointWriter::writeTestAcousticXZ(para, lev, t);
        ////////////////////////////////////////////////////////////////////////
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //t_prev = t;
    ////////////////////////////////////////////////////////////////////////
    //calculate 2nd, 3rd and higher order moments
    ////////////////////////////////////////////////////////////////////////
    if (para->getCalc2ndOrderMoments())  calc2ndMoments(para.get(), cudaMemoryManager.get());
    if (para->getCalc3rdOrderMoments())  calc3rdMoments(para.get(), cudaMemoryManager.get());
    if (para->getCalcHighOrderMoments()) calcHigherOrderMoments(para.get(), cudaMemoryManager.get());
    ////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    //calculate mean on host
    ////////////////////////////////////////////////////////////////////////
    if (para->getCalcMean() && ((int)timestep > para->getTimeCalcMedStart()) && ((int)timestep <= para->getTimeCalcMedEnd()) && ((timestep%(unsigned int)para->getclockCycleForMeasurePoints())==0)) {
        unsigned int tdiff = timestep - previousTimestepForAveraging;
        calcMean(para.get(), tdiff);
        /////////////////////////////////
        //added for incremental averaging
        previousTimestepForAveraging = timestep;
        resetMean(para.get());
        /////////////////////////////////
    }
    if (para->getCalcTurbulenceIntensity()) {
        uint t_diff = timestep - previousTimestepForTurbulenceIntensityCalculation;
        calcTurbulenceIntensity(para.get(), cudaMemoryManager.get(), t_diff);
        //writeAllTiDatafToFile(para.get(), t);
    }
    ////////////////////////////////////////////////////////////////////////
    dataWriter->writeTimestep(para, timestep);
    ////////////////////////////////////////////////////////////////////////
    if (para->getCalcTurbulenceIntensity()) {
        previousTimestepForTurbulenceIntensityCalculation = timestep;
        resetVelocityFluctuationsAndMeans(para.get(), cudaMemoryManager.get());
    }
    ////////////////////////////////////////////////////////////////////////
    if (para->getCalcDragLift()) 
    {
        printDragLift(para.get(), cudaMemoryManager.get(), timestep);
    }
    ////////////////////////////////////////////////////////////////////////

#if DEBUG_FS
    // Write distributions (f's) for debugging purposes.
    DistributionDebugWriter::copyDistributionsToHost(*para, *cudaMemoryManager);
    DistributionDebugWriter::writeDistributions(*para, timestep);
#endif

    ////////////////////////////////////////////////////////////////////////
    VF_LOG_INFO("... done");
    ////////////////////////////////////////////////////////////////////////
}

Simulation::~Simulation()
{
    // CudaFreeHostMemory
    for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
        cudaMemoryManager->cudaFreeCoord(lev);
        cudaMemoryManager->cudaFreeSP(lev);
        if (para->getCalcMean()) {
            cudaMemoryManager->cudaFreeMeanSP(lev);
        }
    }
    if (para->getMaxLevel() > 1) {
        for (int lev = para->getCoarse(); lev < para->getFine(); lev++) {
            cudaMemoryManager->cudaFreeInterfaceCF(lev);
            cudaMemoryManager->cudaFreeInterfaceFC(lev);
            cudaMemoryManager->cudaFreeInterfaceOffCF(lev);
            cudaMemoryManager->cudaFreeInterfaceOffFC(lev);
        }
    }


    //Boundary conditions
    for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
        cudaMemoryManager->cudaFreeNoSlipBC(lev);
        cudaMemoryManager->cudaFreeSlipBC(lev);
        cudaMemoryManager->cudaFreeVeloBC(lev);
        cudaMemoryManager->cudaFreeGeomBC(lev);
        cudaMemoryManager->cudaFreeStressBC(lev);
        cudaMemoryManager->cudaFreePrecursorBC(lev);
        cudaMemoryManager->cudaFreeOutflowBC(lev);
        cudaMemoryManager->cudaFreeDirectionalBoundaryCondition(lev);
    }

    //////////////////////////////////////////////////////////////////////////
    // Temp
    if (para->getDiffOn()) {
        for (int lev = para->getCoarse(); lev < para->getFine(); lev++) {
            cudaMemoryManager->cudaFreeConcentration(lev);
            cudaMemoryManager->cudaFreeConcentrationNoFluxBC(lev);
            cudaMemoryManager->cudaFreeConcentrationFluxBC(lev);
            cudaMemoryManager->cudaFreeConcentrationDirichletBC(lev);
            cudaMemoryManager->cudaFreeConcentrationNeumannBC(lev);
            cudaMemoryManager->cudaFreeSurfaceLayerBC(lev);
            cudaMemoryManager->cudaFreeDirectionalADBoundaryCondition(lev);

            if(para->getUseTurbulentDiffusivity())
                cudaMemoryManager->cudaFreeTurbulentDiffusivity(lev);
            if(para->getBuoyancyEnabled())
                cudaMemoryManager->cudaFreeLocalReferenceTemperature(lev);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // free second order moments
    if (para->getCalc2ndOrderMoments()) {
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            cudaMemoryManager->cudaFree2ndMoments(lev);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // free third order moments
    if (para->getCalc3rdOrderMoments()) {
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            cudaMemoryManager->cudaFree3rdMoments(lev);
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // free higher order moments
    if (para->getCalcHighOrderMoments()) {
        for (int lev = para->getCoarse(); lev <= para->getFine(); lev++) {
            cudaMemoryManager->cudaFreeHigherMoments(lev);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    //Multi GPU
    //////////////////////////////////////////////////////////////////////////
    //3D domain decomposition
    if (para->getNumprocs() > 1) {
        for (int lev = para->getCoarse(); lev < para->getFine(); lev++) {
            auto& parH = para->getParHostAsReference(lev);
            auto& parD = para->getParDeviceAsReference(lev);
            //////////////////////////////////////////////////////////////////////////
            for (uint i = 0; i < para->getNumberOfProcessNeighborsX(lev, "send"); i++) {
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.sendProcessNeighborsX[i], parD.sendProcessNeighborsX[i]);
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.recvProcessNeighborsX[i], parD.recvProcessNeighborsX[i]);
            }
            //////////////////////////////////////////////////////////////////////////
            for (uint i = 0; i < para->getNumberOfProcessNeighborsY(lev, "send"); i++) {
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.sendProcessNeighborsY[i], parD.sendProcessNeighborsY[i]);
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.recvProcessNeighborsY[i], parD.recvProcessNeighborsY[i]);
            }
            //////////////////////////////////////////////////////////////////////////
            for (uint i = 0; i < para->getNumberOfProcessNeighborsZ(lev, "send"); i++) {
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.sendProcessNeighborsZ[i], parD.sendProcessNeighborsZ[i]);
                cudaMemoryManager->cudaFreeProcessNeighbor(parH.recvProcessNeighborsZ[i], parD.recvProcessNeighborsZ[i]);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////
    // Turbulence Intensity
    if (para->getCalcTurbulenceIntensity()) {
        cudaFreeTurbulenceIntensityArrays(para.get(), cudaMemoryManager.get());
    }

        // Cuda Streams
    if (para->getUseStreams()) {
        para->getStreamManager()->destroyCudaEvents();
        para->getStreamManager()->terminateStreams();
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::getVelocityBCCoordinates(int level, real* coords_out, uint n_bc) const
{
    if (!para) {
        throw std::runtime_error("Simulation::getVelocityBCCoordinates: Parameter object is null");
    }

    uint n_bc_actual = para->getParH(level)->velocityBC.numberOfBCnodes;
    if (n_bc != n_bc_actual) {
        throw std::runtime_error("Simulation::getVelocityBCCoordinates: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));
    }

    for (uint i = 0; i < n_bc; i++) {
        int node_idx = para->getParH(level)->velocityBC.k[i];
        coords_out[i * 3 + 0] = para->getParH(level)->coordinateX[node_idx];
        coords_out[i * 3 + 1] = para->getParH(level)->coordinateY[node_idx];
        coords_out[i * 3 + 2] = para->getParH(level)->coordinateZ[node_idx];
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::updateVelocityBC(int level, const real* u, const real* v, const real* w, uint n_bc)
{
    if (!para || !cudaMemoryManager) {
        throw std::runtime_error("Simulation::updateVelocityBC: Parameter or CudaMemoryManager is null");
    }

    uint n_bc_actual = para->getParH(level)->velocityBC.numberOfBCnodes;
    if (n_bc != n_bc_actual) {
        throw std::runtime_error("Simulation::updateVelocityBC: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));
    }

    real velocityRatio = para->getVelocityRatio();

    for (uint i = 0; i < n_bc; i++) {
        para->getParH(level)->velocityBC.Vx[i] = u[i] / velocityRatio;
        para->getParH(level)->velocityBC.Vy[i] = v[i] / velocityRatio;
        para->getParH(level)->velocityBC.Vz[i] = w[i] / velocityRatio;
    }

    cudaMemoryManager->cudaCopyVeloBC(level);
}

//////////////////////////////////////////////////////////////////////////
void Simulation::getSurfaceLayerBCCoordinates(int level, real* coords_out, uint n_bc) const
{
    if (!para) {
        throw std::runtime_error("Simulation::getSurfaceLayerBCCoordinates: Parameter object is null");
    }

    uint n_bc_actual = para->getParH(level)->surfaceLayerBC.numberOfBCnodes;
    if (n_bc != n_bc_actual) {
        throw std::runtime_error("Simulation::getSurfaceLayerBCCoordinates: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));
    }

    for (uint i = 0; i < n_bc; i++) {
        int node_idx = para->getParH(level)->surfaceLayerBC.k[i];
        coords_out[i * 3 + 0] = para->getParH(level)->coordinateX[node_idx];
        coords_out[i * 3 + 1] = para->getParH(level)->coordinateY[node_idx];
        coords_out[i * 3 + 2] = para->getParH(level)->coordinateZ[node_idx];
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::updateSurfaceLayerHeatFlux(int level, const real* hfx, uint n_bc)
{
    if (!para || !cudaMemoryManager) {
        throw std::runtime_error("Simulation::updateSurfaceLayerHeatFlux: Parameter or CudaMemoryManager is null");
    }

    uint n_bc_actual = para->getParH(level)->surfaceLayerBC.numberOfBCnodes;
    if (n_bc != n_bc_actual) {
        throw std::runtime_error("Simulation::updateSurfaceLayerHeatFlux: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));
    }

    for (uint i = 0; i < n_bc; i++) {
        para->getParH(level)->surfaceLayerWallModel.temperatureParameters.surfaceHeatFlux[i] = hfx[i];
    }

    cudaMemoryManager->cudaCopyTemperatureWallModel(
        para->getParH(level)->surfaceLayerWallModel.temperatureParameters,
        para->getParD(level)->surfaceLayerWallModel.temperatureParameters,
        para->getParH(level)->surfaceLayerBC.numberOfBCnodes);
}

//////////////////////////////////////////////////////////////////////////
void Simulation::getADDirichletBCCoordinates(int level, real* coords_out, uint n_bc) const
{
    if (!para)
        throw std::runtime_error("Simulation::getADDirichletBCCoordinates: Parameter object is null");

    const auto& bc = para->getParH(level)->AdvectionDiffusionDirichletBC;
    if (n_bc != (uint)bc.numberOfBCnodes)
        throw std::runtime_error("Simulation::getADDirichletBCCoordinates: Size mismatch. Expected " +
                                 std::to_string(bc.numberOfBCnodes) + " but got " + std::to_string(n_bc));

    for (uint i = 0; i < n_bc; i++) {
        const int node_idx = bc.BCNodeIndices[i];
        coords_out[i * 3 + 0] = para->getParH(level)->coordinateX[node_idx];
        coords_out[i * 3 + 1] = para->getParH(level)->coordinateY[node_idx];
        coords_out[i * 3 + 2] = para->getParH(level)->coordinateZ[node_idx];
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::updateADDirichletBC(int level, const real* concentration, uint n_bc)
{
    if (!para || !cudaMemoryManager)
        throw std::runtime_error("Simulation::updateADDirichletBC: Parameter or CudaMemoryManager is null");

    auto& bc = para->getParH(level)->AdvectionDiffusionDirichletBC;
    if (n_bc != (uint)bc.numberOfBCnodes)
        throw std::runtime_error("Simulation::updateADDirichletBC: Size mismatch. Expected " +
                                 std::to_string(bc.numberOfBCnodes) + " but got " + std::to_string(n_bc));

    // concentration is the temperature deviation from the reference temperature (K), no unit scaling
    for (uint i = 0; i < n_bc; i++)
        bc.concentration[i] = concentration[i];

    cudaMemoryManager->cudaCopyConcentrationDirichletBCHostToDevice(level);
}

//////////////////////////////////////////////////////////////////////////
uint Simulation::setCellPerturbationCells(int level, real x0, real y0, real z0, real cellDx, real cellDy, real dz,
                                          const int* cellTable, uint nxc, uint nyc, uint nzc, uint numberOfCells)
{
    if (!para)
        throw std::runtime_error("Simulation::setCellPerturbationCells: Parameter object is null");
    cellPerturbation = std::make_unique<CellPerturbation>(para.get(), level, x0, y0, z0, cellDx, cellDy, dz, cellTable,
                                                          nxc, nyc, nzc, numberOfCells);
    return cellPerturbation->getNumberOfNodes();
}

//////////////////////////////////////////////////////////////////////////
void Simulation::applyCellPerturbation(const real* thetaPrime, const real* uPrime, const real* vPrime)
{
    if (!cellPerturbation)
        throw std::runtime_error("Simulation::applyCellPerturbation: call setCellPerturbationCells first");
    cellPerturbation->apply(para.get(), thetaPrime, uPrime, vPrime);
}

//////////////////////////////////////////////////////////////////////////
uint Simulation::getNumberOfPerturbationCells() const
{
    return cellPerturbation ? cellPerturbation->getNumberOfCells() : 0;
}

//////////////////////////////////////////////////////////////////////////
uint Simulation::setRelaxationZone(int level, real xMin, real xMax, real yMin, real yMax, real zMax, real width,
                                   real topWidth, uint faces)
{
    if (!para)
        throw std::runtime_error("Simulation::setRelaxationZone: Parameter object is null");
    relaxationZone =
        std::make_unique<BoundaryRelaxationZone>(para.get(), level, xMin, xMax, yMin, yMax, zMax, width, topWidth, faces);
    return relaxationZone->getNumberOfNodes();
}

//////////////////////////////////////////////////////////////////////////
uint Simulation::getNumberOfRelaxationZoneNodes() const
{
    return relaxationZone ? relaxationZone->getNumberOfNodes() : 0;
}

//////////////////////////////////////////////////////////////////////////
void Simulation::getRelaxationZoneNodes(real* coords_out, real* weights_out) const
{
    if (!relaxationZone)
        throw std::runtime_error("Simulation::getRelaxationZoneNodes: call setRelaxationZone first");
    const auto& x = relaxationZone->getCoordinatesX();
    const auto& y = relaxationZone->getCoordinatesY();
    const auto& z = relaxationZone->getCoordinatesZ();
    const auto& w = relaxationZone->getWeights();
    for (size_t i = 0; i < x.size(); i++) {
        coords_out[i * 3 + 0] = x[i];
        coords_out[i * 3 + 1] = y[i];
        coords_out[i * 3 + 2] = z[i];
        weights_out[i] = w[i];
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::pushRelaxationTargets(const real* vx, const real* vy, const real* vz, const real* theta)
{
    if (!relaxationZone)
        throw std::runtime_error("Simulation::pushRelaxationTargets: call setRelaxationZone first");
    relaxationZone->pushTargets(vx, vy, vz, theta);
}

//////////////////////////////////////////////////////////////////////////
void Simulation::applyRelaxation(real alpha, real rate, bool relaxVelocity, bool relaxTemperature)
{
    if (!relaxationZone)
        throw std::runtime_error("Simulation::applyRelaxation: call setRelaxationZone first");
    relaxationZone->apply(para.get(), alpha, rate, relaxVelocity, relaxTemperature);
}

//////////////////////////////////////////////////////////////////////////
void Simulation::getFreestreamBCCoordinates(int level, real* coords_out, uint n_bc) const
{
    if (!para)
        throw std::runtime_error("Simulation::getFreestreamBCCoordinates: Parameter object is null");

    uint n_bc_actual = para->getParH(level)->freestreamBC.numberOfBCnodes;
    if (n_bc != n_bc_actual)
        throw std::runtime_error("Simulation::getFreestreamBCCoordinates: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));

    for (uint i = 0; i < n_bc; i++) {
        int node_idx = para->getParH(level)->freestreamBC.k[i];
        coords_out[i * 3 + 0] = para->getParH(level)->coordinateX[node_idx];
        coords_out[i * 3 + 1] = para->getParH(level)->coordinateY[node_idx];
        coords_out[i * 3 + 2] = para->getParH(level)->coordinateZ[node_idx];
    }
}

//////////////////////////////////////////////////////////////////////////
void Simulation::updateFreestreamBC(int level, const real* u, const real* v, const real* w, uint n_bc)
{
    if (!para || !cudaMemoryManager)
        throw std::runtime_error("Simulation::updateFreestreamBC: Parameter or CudaMemoryManager is null");

    uint n_bc_actual = para->getParH(level)->freestreamBC.numberOfBCnodes;
    if (n_bc != n_bc_actual)
        throw std::runtime_error("Simulation::updateFreestreamBC: Size mismatch. Expected " +
                               std::to_string(n_bc_actual) + " but got " + std::to_string(n_bc));

    real velocityRatio = para->getVelocityRatio();
    for (uint i = 0; i < n_bc; i++) {
        para->getParH(level)->freestreamBC.Vx[i] = u[i] / velocityRatio;
        para->getParH(level)->freestreamBC.Vy[i] = v[i] / velocityRatio;
        para->getParH(level)->freestreamBC.Vz[i] = w[i] / velocityRatio;
    }

    cudaMemoryManager->cudaCopyFreestreamBC(level);
}

}

//! \}
