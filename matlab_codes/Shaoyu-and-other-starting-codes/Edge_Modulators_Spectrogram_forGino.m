% This routine calculates the RMS of the Pre and Probe pulse signals
%
% Written by Shaoyu Qiao, March 26, 2019 

clear all;
close all

subjects = {'maverick','archie'};

for iSubject = 1% : length(subjects) % Loop on the animals
    clearvars -except subjects iSubject
    if strcmp(subjects{iSubject},'archie')
        archie_vSUBNETS220_rig3
    else
        maverick_vSUBNETS220_rig3
    end
    PreStimSess = PreStimResponseAll_Database_NetworkEdge;
    
    for iSess = 1 : numel(PreStimSess)
        %         pAccLLRperm = PreStimSess{iSess}{end-2};
        %         useSessIndx(iSess) = pAccLLRperm <= 0.05 & ~isnan(pAccLLRperm);
        
        pAccLLRperm = PreStimSess{iSess}{end-2};
        pFDR_logic = PreStimSess{iSess}{end-1};
        useSessIndx(iSess) = pFDR_logic & ~isnan(pAccLLRperm);
    end
    
    UsedSess = find(useSessIndx);
    
    for iSess = 15%UsedSess
        clearvars -except iSess PreStimSess DATADIR FIGUREDIR MONKEYDIR iSubject subjects UsedSess
        
        disp(['Session ' num2str(iSess) ' out of ' num2str(length(PreStimSess)) ' ...'])
        
        RespPair = sessElectrode(PreStimSess{iSess}); % responding channel
        stimName = PreStimSess{iSess}{9};
        stimTask = PreStimSess{iSess}{7};
        day = sessDay(PreStimSess{iSess});
        
        %% loading Pre data        
        dataDir_Pre = sprintf('%s/AccLLR/%sStimAllSess/StimResponseSessions/',DATADIR,stimName);
        switch stimTask
            case 'StimSinglePulse'
                fileName_Pre = sprintf('%sSess%03d_%s_AccLLR_Elec%03d-Elec%03d_%s_1stPulse.mat',dataDir_Pre,iSess,day,RespPair(1),RespPair(2),stimName);
                
            case 'StimBlockShort'
                fileName_Pre = sprintf('%sSess%03d_%s_AccLLR_Elec%03d-Elec%03d_%s_grouped.mat',dataDir_Pre,iSess,day,RespPair(1),RespPair(2),stimName);
        end
        
        tic
        disp('Loading Pre data ...')
        load(fileName_Pre)
        disp('Done with Pre data loading')
        toc
        
        %% identifying beta modulators %%%%%%
        fs = Data.Fs.lfp;% lfp sampling rate
        Fs = Data.Fs.raw;% raw sampling rate
        
        AnalParams = Data.Params.Anal;
        AnalParams.Tapers = [0.5,2];
        AnalParams.TestSpecDiff.fk = [10 40];
        
        fkNames = {'\beta'};
        Data.Params.Anal = AnalParams;
        Data.Spec.ROC.fk = AnalParams.TestSpecDiff.fk;
        StimTrials = Data.StimTrials(Data.goodTrials_index);
        sys = StimTrials(1).MT;
        bn_Pre = [-1005 -5]; % ms
        
        %% extract AccLLR results
        Results = Data.AccLLR.Results;
        EventST = Results.EventST;
        
        TargCh = Results.Ch;
        LPRawEvent = Results.LPRawEvent;
        LPRawNull = Results.LPRawNull;
        StimArtifactBlankWin = Results.StimArtifactBlankWin; % ms
        StimArtifactStartInd = Results.StimArtifactStartInd;
        AccLLRwin = Results.AccLLRwin;
        nTr = length(Results.EventST);
        AccLLRRawNull = Results.NoHist.Null.AccLLR;
        mAccLLRRawNull = mean(AccLLRRawNull);
        
        [dum,ind] = sort(EventST);
        nTr_CorrectDetect = sum(~isnan(dum));
        
        mLPRawEvent = mean(LPRawEvent,1);
        if mLPRawEvent(round(mean(EventST(~isnan(EventST))))) > 0
            LPRawEvent = -LPRawEvent;
        end
        
        if ~isequal(sum(~isnan(EventST)),0) % if detected
            nFreqBands = size(AnalParams.TestSpecDiff.fk,1);
            
            if strcmp(Data.Spec.recordConfig,'Bipolar')
                lfp_Detected = Data.spec.lfp.bipolar.Detected;
                lfp_notDetected = Data.spec.lfp.bipolar.notDetected;
                
                Lfp_Pre = [];
                for iFreqBand = 1 : nFreqBands
                    pval = [];
                    pval_roc = [];
                    AnalParams.Spec.Test.fk = AnalParams.TestSpecDiff.fk(iFreqBand,:);
                    Fk = AnalParams.Spec.Test.fk;
                    
                    if ~isempty(Data.Spec.ROC.sigChs{iFreqBand})
                        % load LFPs
                        if isempty(Lfp_Pre)
                            try
                                disp('Loading lfp data ...')
                                [Lfp_Pre] = trialStimPulseLfp(StimTrials, sys, [], [], 'PulseStarts', bn_Pre); % returns monopolar recording
                                
                            catch
                                disp('Loading raw data ...')
                                [Raw_Pre] = trialStimPulseRaw(StimTrials, sys, [], [], 'PulseStarts', bn_Pre); % returns monopolar recording
                                
                                for iCh = 1 : size(Raw_Pre,2)
                                    lfp_Pre(:,iCh,:) = mtfilter(sq(Raw_Pre(:,iCh,:)),[0.0025,400],Fs,0);
                                end
                                Lfp_Pre = lfp_Pre(:,:,1:Fs/fs:end);
                                clear Raw_Pre lfp_Pre
                            end
                        end
                        
                        fprintf('\n');
                        disp(['Start ROC on PSDs @ ' num2str(AnalParams.TestSpecDiff.fk(iFreqBand,1)) '-' num2str(AnalParams.TestSpecDiff.fk(iFreqBand,2)) ' Hz'])
                        fprintf('\n\n');
                        reverseStr = '';
                        
                        % get modulator channel
%                         chs_FreqBand = Data.Spec.ROC.sigChs{iFreqBand};
%                         nCh_Use = numel(chs_FreqBand);
                        keyboard
                        %nCh_Use = size(Data.RecordPair,1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             se = size(Data.RecordPair,1)
                        for iCh = 1 %: nCh_Use
                            
                            %iCh = find(chs_FreqBand(indx)==Data.RecordPair(:,1));
                            
%                             fprintf('\n\n');
%                             msg = sprintf('%d/%d Channels',iCh,nCh_Use);
%                             fprintf([reverseStr,msg]);
%                             reverseStr = repmat(sprintf('\b'),1,length(msg));
%                             fprintf('\n\n');
%                             
%                             stdThresh = Data.spec.lfp.stdThresh;
%                             X1 = sq(lfp_Detected(:,iCh,:));
%                             [X1,goodInd1] = removeNoisyLfpTrials(X1,stdThresh);
%                             
%                             X2 = sq(lfp_notDetected(:,iCh,:));
%                             [X2,goodInd2] = removeNoisyLfpTrials(X2,stdThresh);
%                             
%                             
%                             figure('Position',[100 100 2000 800])
%                             subplot(2,7,1)
%                             plotElectrodeLocationMRIoverlay(day,TargCh(1));
%                             title(['Receiver e' num2str(TargCh(1))])
%                             
%                             subplot(2,7,8)
%                             plotElectrodeLocationMRIoverlay(day,chs_FreqBand(indx));
%                             title(['Modulator e' num2str(chs_FreqBand(indx))])
%                             
%                             h1 = subplot(2,7,4);
%                             pos1 = get(h1,'Position');
%                             ERPplotRange = [-round(max(abs(LPRawEvent(:))),-1) round(max(abs(LPRawEvent(:))),-1)];
%                             %ERPplotRange = [-30 30];
%                             imagesc((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin,1:nTr,LPRawEvent(ind,:),ERPplotRange);
%                             box off;
%                             xlabel('Time after stim onset (ms)');
%                             ylabel('Sortetd events')
%                             xlim([0 StimArtifactBlankWin+AccLLRwin])
%                             %                             c = colorbar;
%                             %                             c.Label.String = 'Voltage (\muV)';
%                             
%                             colormap(gca,polarmap)
%                             for i = 1 : nTr_CorrectDetect
%                                 text(dum(i)/fs*1e3+StimArtifactBlankWin,i-0.5,'x','color','k');
%                             end
%                             title('Receiver(AccLLR)')
%                             
%                             hold on
%                             h = line([0 size(LPRawEvent,2)/fs*1e3+StimArtifactBlankWin],[nTr_CorrectDetect+0.5 nTr_CorrectDetect+0.5]);
%                             set(h,'Linewidth',1,'Linestyle','--','Color','k');
%                             
%                             text(3,mean(1:nTr_CorrectDetect)+2,'Hit','Rotation',90);
%                             text(3,mean(nTr_CorrectDetect+1:nTr)+3,'Miss','Rotation',90);
%                             
%                             
%                             %% modulator decoder hit vs miss
%                             % plot ROC curve of modulator activity in hit vs miss events
%                             subplot(2,7,2)
%                             [auc,se,S1,S2,roc_Thresh,maxYoudenIndex] = calcRocSpecDiff_HistAUC(X1,X2,AnalParams);
%                             
%                             S_all = [];
%                             
%                             S_all(hitIndx) = S1;
%                             S_all(missIndx) = S2;
%                             
%                             % get different decoding rates across different ROC thresholds
%                             [modDecodeHitRate,modDecodeMissRate,rocThresh,~] = runModulatorDecoder(S1,S2,S_all,EventST,roc_Thresh);
%                                                         
%                             %find optimal decoding rate
%                             ind = modDecodeHitRate > modDecodeMissRate;
%                             if isequal(sum(ind),0) % modulator hit decoding rates across roc thresholds are all greater than miss decoding rate
%                                 iThresh = find(roc_Thresh==maxYoudenIndex);
%                             elseif strcmp(subjects{iSubject},'archie') && iSess == 13
%                                 [~,iThresh] = min(abs(roc_Thresh-4.277));
%                             else
%                                 if ~isequal(sum(modDecodeMissRate>0.5),0)
%                                     indd = find(ind & modDecodeMissRate>0.5);
%                                     modDecodeHitPlusMissRate = modDecodeHitRate(ind & modDecodeMissRate>0.5) + modDecodeMissRate(ind & modDecodeMissRate>0.5);
%                                 else
%                                     indd = find(ind);
%                                     modDecodeHitPlusMissRate = modDecodeHitRate(ind) + modDecodeMissRate(ind);
%                                 end
%                                 [~,modDecodeHitPlusMissRateInd] = max(modDecodeHitPlusMissRate);
%                                 [~,iThresh] = min(abs(roc_Thresh-rocThresh(indd(modDecodeHitPlusMissRateInd))));
%                             end
% 
%                             %iThresh = find(roc_Thresh==maxYoudenIndex);
%                             [optModDecodeHitRate,optModDecodeMissRate,optRocThresh,DecoderTrials] = runModulatorDecoder(S1,S2,S_all,EventST,roc_Thresh,iThresh);
%                             
%                             subplot(2,7,3)
%                             hg1=histogram(S1,'BinMethod','fd','Normalization','count','FaceColor','r');hold on
%                             hg2=histogram(S2,'BinMethod','fd','Normalization','count','FaceColor','k');
%                             plot([optRocThresh,optRocThresh],[0,max([hg1.Values hg2.Values])+1],'b--')
%                             legend('Hit','Miss')
%                             xlabel(['Mean log ' fkNames{iFreqBand} ' power']);
%                             ylabel('Count')
%                             title('Modulator')
%                             xlim([floor(min([S1;S2])),ceil(max([S1;S2]))])
%                             
%                             
%                             sorted_ModHitInRecHit_tr = DecoderTrials.sorted_ModHitInRecHit_tr;
%                             ModHitNotInRecHit_tr = DecoderTrials.ModHitNotInRecHit_tr;
%                             sorted_ModMissInRecHit_tr = DecoderTrials.sorted_ModMissInRecHit_tr;
%                             ModMissNotInRecHit_tr = DecoderTrials.ModMissNotInRecHit_tr;
%                             modulatorDecoder_Hit_tr = DecoderTrials.modulatorDecoder_Hit_tr;
%                             modulatorDecoder_Miss_tr = DecoderTrials.modulatorDecoder_Miss_tr;
%                             dum1 = DecoderTrials.dum1;
%                             dum2 = DecoderTrials.dum2;
%                             
%                             indNew = [];
%                             indNew = [sorted_ModHitInRecHit_tr,ModHitNotInRecHit_tr,sorted_ModMissInRecHit_tr,ModMissNotInRecHit_tr];
%                             
%                             subplot(2,7,[9 10])
%                             plot(rocThresh,modDecodeHitRate','linewidth',2,'color','r'); hold on
%                             plot(rocThresh,modDecodeMissRate','linewidth',2,'color','k');
%                             plot(rocThresh,modDecodeHitRate+modDecodeMissRate,'linewidth',2,'color','g')
%                             plot([optRocThresh,optRocThresh],[floor(min([modDecodeHitRate modDecodeMissRate])*10)/10,ceil(max([modDecodeHitRate modDecodeMissRate])*10)/10],'b--')
%                             xlabel(['Mean log ' fkNames{iFreqBand} ' power']);
%                             ylabel('Decoding accuracy')
%                             ylim([floor(min([modDecodeHitRate modDecodeMissRate])*10)/10,ceil(max([modDecodeHitRate modDecodeMissRate])*10)/10])
%                             legend('Hit','Miss','Location','NorthWest')
%                             title('Modulator ROC decoding')
%                             
%                             
%                             h2 = subplot(2,7,11);
%                             %ERPplotRange = [-round(max(abs(LPRawEvent(:))),-1) round(max(abs(LPRawEvent(:))),-1)];
%                             imagesc((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin,1:length(indNew),LPRawEvent(indNew,:),ERPplotRange);
%                             box off;
%                             xlabel('Time after stim onset (ms)');
%                             %ylabel('Sortetd events')
%                             xlim([0 StimArtifactBlankWin+AccLLRwin])
%                             c = colorbar;
%                             c.Label.String = 'Voltage (\muV)';
%                             colormap(gca,polarmap)
%                             title('Receiver (Decoder)')
%                             
%                             hold on
%                             
%                             h = line([0 size(LPRawEvent,2)/fs*1e3+StimArtifactBlankWin],[numel(modulatorDecoder_Hit_tr)+0.5 numel(modulatorDecoder_Hit_tr)+0.5]);
%                             set(h,'Linewidth',1,'Linestyle','--','Color','k');
%                             
%                             for i = 1 : numel(sorted_ModHitInRecHit_tr)
%                                 text(dum1(i)/fs*1e3+StimArtifactBlankWin,i-0.5,'x','color','k');
%                             end
%                             
%                             for i = numel(modulatorDecoder_Hit_tr)+1 : numel(modulatorDecoder_Hit_tr)+numel(sorted_ModMissInRecHit_tr)
%                                 text(dum2(i-numel(modulatorDecoder_Hit_tr))/fs*1e3+StimArtifactBlankWin,i-0.5,'x','color','k');
%                             end
%                             
%                             text(8,mean(1:numel(modulatorDecoder_Hit_tr))+15,{'Hit';num2str(optModDecodeHitRate)},'Rotation',90)
%                             text(8,mean(numel(modulatorDecoder_Hit_tr)+1:length(indNew))+15,{'Miss';num2str(optModDecodeMissRate)},'Rotation',90)
%                             
%                             pos2 = get(h2,'Position');
%                             set(h2,'Position',[pos2(1),pos2(2),pos1(3),pos1(4)]);
%                             %
%                             
%                             % receiver hit amd miss ERPs
%                             subplot(2,7,6)
%                             avgERP_RecHit_tr = mean(LPRawEvent(hitIndx,:));
%                             semERP_RecHit_tr = std(LPRawEvent(hitIndx,:))/sqrt(numel(hitIndx));
%                             plotSTA((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin, avgERP_RecHit_tr,semERP_RecHit_tr,[1 0 0]);
%                             
%                             hold on
%                             avgERP_RecMiss_tr = mean(LPRawEvent(missIndx,:));
%                             semERP_RecMiss_tr = std(LPRawEvent(missIndx,:))/sqrt(numel(missIndx));
%                             plotSTA((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin,avgERP_RecMiss_tr,semERP_RecMiss_tr,[0 0 0]);
%                             xlabel('Time after stim onset (ms)');
%                             ylabel('Amplitude (\muV)')
%                             title('ERP')
%                             %legend('Hit',' ', 'Miss',' ','Location','SouthEast')
%                             xlim([0 StimArtifactBlankWin+AccLLRwin])
%                             %ylim([-20 10])
%                             
%                             % modulator decoded receiver hit amd miss ERPs
%                             h1 = subplot(2,7,13);
%                             avgERP_ModHit_tr = mean(LPRawEvent(modulatorDecoder_Hit_tr,:));
%                             semERP_ModHit_tr = std(LPRawEvent(modulatorDecoder_Hit_tr,:))/sqrt(numel(modulatorDecoder_Hit_tr));
%                             plotSTA((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin, avgERP_ModHit_tr,semERP_ModHit_tr,[1 0 0]);
%                             
%                             hold on
%                             avgERP_ModMiss_tr = mean(LPRawEvent(modulatorDecoder_Miss_tr,:));
%                             semERP_ModMiss_tr = std(LPRawEvent(modulatorDecoder_Miss_tr,:))/sqrt(numel(modulatorDecoder_Miss_tr));
%                             plotSTA((1:size(LPRawEvent,2))/fs*1e3+StimArtifactBlankWin,avgERP_ModMiss_tr,semERP_ModMiss_tr,[0 0 0]);
%                             xlabel('Time after stim onset (ms)');
%                             ylabel('Amplitude (\muV)')
%                             title('Decoded ERP')
%                             xlim([0 StimArtifactBlankWin+AccLLRwin])
%                             %legend('Decoder Hit',' ', 'Decoder Miss',' ')
%                             pos1 = get(h1,'Position');
                            
                            
                            keyboard  
                            %% %%%%%% Spectrogram, permutation test   GINO %%%%%%
                            hitIndx = Data.spec.lfp.DetectedIndx{iCh}; % labels for the hits (which trial was a hit)
                            missIndx = Data.spec.lfp.notDetectedIndx{iCh}; % labels for the misses (which trial was a miss)
                            ModulatorPair = Data.RecordPair(iCh,:); % get the modulator pair
                            ModulatorLFP_Pre = sq(Lfp_Pre(:,ModulatorPair(1),:) - Lfp_Pre(:,ModulatorPair(2),:));
                            
                            % Parameters
                            tapers = [0.5 5];
                            fk = 60;
                            pad = Data.Params.Anal.pad; % pad = 2
                            fs = Data.Fs.lfp;
                            dn = 0.005;
                            nPerm = 10000;
                            alpha = 0.05;
                            
                            Spec = []; TestSpec = []; DiffSpec = []; AvDiffSpec = [];
                            [Spec,f] = tfspec(ModulatorLFP_Pre(hitIndx,:),tapers,fs,dn,fk,pad); % get the time-frequency spectrum of the hit trial
                            
                            keyboard
                            % zScore spectrogram based on AccLLR
                            % h3 = subplot(2,7,7); G: commented out
                            plotFieldzScoreBipolarSpectrogramGino(ModulatorLFP_Pre,hitIndx,missIndx,bn_Pre,tapers,fs,dn,fk,pad,nPerm,alpha)
                            
                            keyboard
                            
                            c = colorbar;
                            c.Label.String = 'Hit - Miss (Z-score), AccLLR';
                            pos3 = get(h3,'Position');
                            set(h3,'Position',[pos3(1),pos3(2),pos1(3),pos3(4)]);
                            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            
                            
                            
                           
                            warning off
                            keyboard
                            %% saving the figure %%%
                            StimSyllablePair = Data.StimPairs.Syllable;
                            StimResPair = Data.StimResPairs;
                            figDir = [FIGUREDIR '/ModulationNetwork/' stimTask '/PreBaseline/'];
                            figDirBase = sprintf('Sess%03d_%s_StimPathway_Stim%03d-%03d_Rec%03d-%03d',iSess,day,StimSyllablePair(1),StimSyllablePair(2),StimResPair(1),StimResPair(2));
                            FigDir = [figDir figDirBase '/'];
                            if ~exist(FigDir,'dir')
                                mkdir(FigDir)
                            end
                            set(gcf,'PaperPositionMode','auto')
                            print(sprintf('%sSpec_Sess%03d_%s_Modulator_e%03d_e%03d_%02d_%02dHz_ROC.svg',FigDir,iSess,day,ModulatorPair(1),ModulatorPair(2),Fk(1),Fk(2)),'-dsvg')
                            print(sprintf('%sSpec_Sess%03d_%s_Modulator_e%03d_e%03d_%02d_%02dHz_ROC.png',FigDir,iSess,day,ModulatorPair(1),ModulatorPair(2),Fk(1),Fk(2)),'-dpng')
                            %%%%%%
                            close
                        end
                    end
                end
            end
        end
        clear Data
        
    end
end