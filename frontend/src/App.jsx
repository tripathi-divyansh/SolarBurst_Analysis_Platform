import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import Navigation from './components/Navigation';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import UploadInspect from './pages/UploadInspect';
import LightCurveExplorer from './pages/LightCurveExplorer';
import DetectionControls from './pages/DetectionControls';
import BurstCatalog from './pages/BurstCatalog';
import EventDetail from './pages/EventDetail';
import WaveletAnalysis from './pages/WaveletAnalysis';
import ModelTraining from './pages/ModelTraining';
import Distributions from './pages/Distributions';
import ExportReport from './pages/ExportReport';
import MethodsHelp from './pages/MethodsHelp';
import { ThemeProvider } from './context/ThemeContext';
import { api } from './api';

export default function App() {
  const [activeDataset, setActiveDataset] = useState(null);
  const [activeJob, setActiveJob] = useState(null);
  const [selectedBurst, setSelectedBurst] = useState(null);
  const [isDemoLoading, setIsDemoLoading] = useState(false);

  // Auto-load existing dataset and results on mount if available
  useEffect(() => {
    async function loadInitial() {
      try {
        const datasets = await api.listDatasets();
        if (datasets && datasets.length > 0) {
          const ds = datasets[0];
          setActiveDataset(ds);
          try {
            const jobs = await api.listJobs();
            const completedJob = jobs.find(j => j.dataset_id === ds.id && j.status === 'completed') || jobs[0];
            if (completedJob) {
              const fullJob = await api.getJob(completedJob.id);
              setActiveJob(fullJob);
              if (fullJob.result_json?.bursts?.length > 0) {
                setSelectedBurst(fullJob.result_json.bursts[0]);
              }
            }
          } catch (err) {
            console.log('[Initial Result Load Note]', err.message);
          }
        }
      } catch (err) {
        console.error('[Initial Datasets Load Error]', err);
      }
    }
    loadInitial();
  }, []);

  // Poll running jobs
  useEffect(() => {
    let interval = null;
    if (activeJob && activeJob.status === 'running') {
      interval = setInterval(async () => {
        try {
          const updated = await api.getJob(activeJob.id);
          setActiveJob(updated);
          if (updated.status === 'completed' && updated.result_json?.bursts?.length > 0) {
            if (!selectedBurst) {
              setSelectedBurst(updated.result_json.bursts[0]);
            }
          }
        } catch (err) {
          console.error('[Job Poll Error]', err);
        }
      }, 1500);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeJob, selectedBurst]);

  // Load demo observation
  const handleLoadDemo = async () => {
    setIsDemoLoading(true);
    try {
      const res = await api.loadDemo();
      setActiveDataset(res.dataset);
      setActiveJob(res.job);
      if (res.job?.result_json?.bursts?.length > 0) {
        setSelectedBurst(res.job.result_json.bursts[0]);
      }
    } catch (err) {
      alert('Error loading demo: ' + err.message);
    } finally {
      setIsDemoLoading(false);
    }
  };

  // Start analysis job
  const handleStartAnalysis = async (datasetId, config) => {
    try {
      const res = await api.startAnalysis(datasetId, config);
      setActiveJob(res.job);
    } catch (err) {
      alert('Error starting analysis: ' + err.message);
    }
  };

  // Cancel running job
  const handleCancelJob = async (jobId) => {
    try {
      await api.cancelJob(jobId);
      const updated = await api.getJob(jobId);
      setActiveJob(updated);
    } catch (err) {
      alert('Error cancelling job: ' + err.message);
    }
  };

  // Update audit status
  const handleUpdateAudit = async (burstId, auditData) => {
    try {
      const res = await api.updateAudit(burstId, auditData);
      // Update burst in local job result state
      if (activeJob?.result_json?.bursts) {
        const updatedBursts = activeJob.result_json.bursts.map((b) =>
          b.id === burstId || `${activeJob.id}_${b.burst_id}` === burstId
            ? { ...b, ...auditData }
            : b
        );
        setActiveJob({
          ...activeJob,
          result_json: {
            ...activeJob.result_json,
            bursts: updatedBursts
          }
        });
        if (selectedBurst && (selectedBurst.id === burstId || `${activeJob.id}_${selectedBurst.burst_id}` === burstId)) {
          setSelectedBurst({ ...selectedBurst, ...auditData });
        }
      }
    } catch (err) {
      alert('Failed to update audit: ' + err.message);
    }
  };

  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="app-layout">
          <Navigation />

          <div className="main-content">
            <Header
              activeDataset={activeDataset}
              activeJob={activeJob}
              onLoadDemo={handleLoadDemo}
              isDemoLoading={isDemoLoading}
            />

            <Routes>
              <Route
                path="/"
                element={
                  <Dashboard
                    activeDataset={activeDataset}
                    activeJob={activeJob}
                    onLoadDemo={handleLoadDemo}
                    isDemoLoading={isDemoLoading}
                  />
                }
              />
              <Route
                path="/upload"
                element={
                  <UploadInspect
                    onDatasetLoaded={setActiveDataset}
                    onStartAnalysis={handleStartAnalysis}
                  />
                }
              />
              <Route
                path="/explorer"
                element={
                  <LightCurveExplorer
                    activeDataset={activeDataset}
                    activeJob={activeJob}
                    selectedBurst={selectedBurst}
                    onSelectBurst={setSelectedBurst}
                  />
                }
              />
              <Route
                path="/controls"
                element={
                  <DetectionControls
                    activeDataset={activeDataset}
                    activeJob={activeJob}
                    onStartAnalysis={handleStartAnalysis}
                    onCancelJob={handleCancelJob}
                  />
                }
              />
              <Route
                path="/catalog"
                element={
                  <BurstCatalog
                    activeJob={activeJob}
                    selectedBurst={selectedBurst}
                    onSelectBurst={setSelectedBurst}
                    onUpdateAudit={handleUpdateAudit}
                  />
                }
              />
              <Route
                path="/detail"
                element={
                  <EventDetail
                    activeJob={activeJob}
                    selectedBurst={selectedBurst}
                    onSelectBurst={setSelectedBurst}
                  />
                }
              />
              <Route
                path="/wavelet"
                element={
                  <WaveletAnalysis
                    activeDataset={activeDataset}
                    selectedBurst={selectedBurst}
                  />
                }
              />
              <Route
                path="/models"
                element={<ModelTraining />}
              />
              <Route
                path="/distributions"
                element={<Distributions activeJob={activeJob} />}
              />
              <Route
                path="/export"
                element={
                  <ExportReport
                    activeDataset={activeDataset}
                    activeJob={activeJob}
                  />
                }
              />
              <Route
                path="/help"
                element={<MethodsHelp />}
              />
              <Route
                path="/methods"
                element={<MethodsHelp />}
              />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
}
