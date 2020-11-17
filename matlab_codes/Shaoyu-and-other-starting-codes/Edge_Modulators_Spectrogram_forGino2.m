% This routine calculates the RMS of the Pre and Probe pulse signals
%
% Written by Shaoyu Qiao, March 26, 2019 

clear all;
close all

subjects = {'maverick','archie'};

%for 
    iSubject = 1% : length(subjects) % Loop on the animals
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
    
    for iSess = 7 %UsedSess
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
                        %nCh_Use = size(Data.RecordPair,1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       se = size(Data.RecordPair,1)
                        for iCh = 1:nCh_Use   
                            
                            display(['Electrode # ......',num2str(iCh)]);
                            
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
                            
                       
                            Hits_Misses_Spectrograms_Gino(ModulatorLFP_Pre,hitIndx,missIndx,bn_Pre,tapers,fs,dn,fk,pad,nPerm,alpha,iSubject,iSess,iCh)                  
             
                        end
                    end
                end
            end
        end
        clear Data
        
    end
%end