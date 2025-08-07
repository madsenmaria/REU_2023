%%%% JUST PLOTTING COMPOSITES 
latmax = 80; % 87.5 N
latmin = 10; % 10 N
lonmax = 300; % easternmost longitude (0 to 360)
lonmin = 100; % westernmost longitude (0 to 360)
% Get the latitudes and longitudes
datalat = nc_varget([dir ext1 int2str(year_end) ext2],'lat');
% Isolate the latitudes in the domain of interest
ind_lat = find(datalat >= latmin & datalat <= latmax);
datalat = datalat(ind_lat);
nlat = length(datalat); % Total number of latitudes

datalon = nc_varget([dir ext1 int2str(year_end) ext2],'lon');



% Isolate the longitudes in the domain of interest.  Here we must be
% careful in the case that the domain crosses the prime meridian because 
% then the value of lonmax would be less than lonmin.  We add
% an if statement to account for this possibility.
if lonmax < lonmin
    
    % Isolate the longitudes in the western part of the domain
    I_west = find(datalon >= lonmin & datalon <= 360);
    % Do the same for the eastern part of the domain
    I_east = find(datalon <= lonmax);

    % Redefine the longitude array
    datalon = datalon([I_west; I_east]);
    
else % the case that the domain does not cross the prime meridian

    ind_lon = find(datalon >= lonmin & datalon <= lonmax);
    datalon = datalon(ind_lon);
end

nlon = length(datalon); % Total number of longitudes
TotGrids = nlat*nlon; % total number of grid points

% Create lat and lon graticules for plotting the patterns in the last part
% of this code
latgrat = repmat(datalat,1,nlon);
longrat = repmat(datalon',nlat,1);
%% Load in data

%load 48to19_2021.mat
%load 74_seasons_SOM.mat

timeseries = timeseries(91:10811, :, :); 
%timeseries = timeseries(91:11113, :, :); 

val = 373; % one standard deviation from timeseires total mean (std is 51.05) 53.73 in new
somtry1 = find(timeseries(:,3) == 1 & timeseries(:,4) <= val) ;
somtry2 = find(timeseries(:,3) == 2 & timeseries(:,4) <= val) ;
somtry3 = find(timeseries(:,3) == 3 & timeseries(:,4) <= val) ;
somtry4 = find(timeseries(:,3) == 4 & timeseries(:,4) <= val) ;
somtry5 = find(timeseries(:,3) == 5 & timeseries(:,4) <= val) ;
somtry6 = find(timeseries(:,3) == 6  & timeseries(:,4) <= val) ;
somtry7 = find(timeseries(:,3) == 7 & timeseries(:,4) <= val) ;
somtry8 = find(timeseries(:,3) == 8 & timeseries(:,4) <= val) ;
somtry9 = find(timeseries(:,3) == 9 & timeseries(:,4) <= val) ;
somtry10 = find(timeseries(:,3) == 10 & timeseries(:,4) <= val) ;
somtry11 = find(timeseries(:,3) == 11 & timeseries(:,4) <= val) ;
somtry12 = find(timeseries(:,3) == 12 & timeseries(:,4) <= val) ;

load composites105.mat
load olr105.mat
load latlon.mat

lat1 = lat; 
 %% FOR LAGS BEFORE AND AFTER
alldays = datenum('1948-11-01'):datenum('2019-3-31') ; 
[yr, mo, day] = datevec(alldays) ; 
kp = find(ismember(mo, [11 12 1:3]) & ~((mo == 2) & (day == 29))) ;


%%  FINDING SPECIFIC TRANSITIONS
lag = 10; 


for i = 5  % start of transition
    for j = 5 % end of transition
        eval(['ind1 = somtry' num2str(i) ' ;'])
        ind = find(ismember(kp(ind1)+lag,kp)); % make sure lag is part of data and doesn't skip seasons
        jkl = ind1(ind) + lag;  
        trans_end = find(timeseries((jkl), 3) == j) ; 
        %trans_end = find(timeseries((jkl), 3) ~= j & timeseries((jkl),4) <= val) ; 
        trans_point = jkl(trans_end);
    end
end
trans_point2 = trans_point-lag; % timeseries value for i that transitions into j
 

 x = [diff(trans_point2')~=1,true];
 comptrans = trans_point2(x);  % Now just use the end of consecutive dates
 % comptrans = trans_point2;
    

%% DEFINING LAGS WITH LOOP cosine weight?


lag = -10:5:10; 

for i = 1:length(lag)
    lagg = lag(i); 
    somnum = comptrans;
    ind = find(ismember(kp(somnum)+lagg, kp)) ; 
    somnum = somnum(ind) + lagg ; 
    
      % calculate 500hgt anomaly
     datdiff(:,:,i) = squeeze(mean(hgt500(somnum,:,:)))- ...  % lat, lon, lag
        squeeze(mean(hgt500(:,:,:)));
     % calculate SLP anomaly
     datdiff_slp(:,:,i) = squeeze(mean(slp(somnum,:,:)))- ...
        squeeze(mean(slp(:,:,:)));
    % calculate u250 anomaly
     datdiff_u250(:,:,i) = squeeze(mean(compwind(somnum,:,:)))- ...
        squeeze(mean(compwind(:,:,:)));
    
    % calculate 850temp anomaly
     temp(:,:,i) = squeeze(mean(temp850(somnum,:,:))); 
     datdiff_850t(:,:,i) = squeeze(mean(temp850(somnum,:,:)))-...
        squeeze(mean(temp850(:,:,:)));
    
    B = 100 * ones(29, 81);


    slp_hpa(:,:,i) = datdiff_slp(:,:,i) ./ B ;
end

%% Cosine weight 
addpath '/Users/mariamadsen/Desktop/Research/EOF_research/daily_uwnd/uwnd/'
addpath '/Users/mariamadsen/Desktop/Research/EOF_research/daily_uwnd/'
lon = ncread('uwnd.1948.nc','lon'); % 144 Longitude points
lat = ncread('uwnd.1948.nc','lat'); % 73 latitude points

 
% Only consider Northern Hemisphere latitudes:
lat_find=find(lat>=10 & lat<=80);
lat_orig=lat;
lat=lat(lat_find);

%Cosine weight hgt500
datdiff = permute(datdiff,[3,1,2]);  % This changes order so we have data, lat, lon
datdiff = cosweight(datdiff, lat) ; 
datdiff = permute(datdiff,[2,3,1]); 
datdiff = double(datdiff); 

%Cosine weight u250
datdiff_u250 = permute(datdiff_u250,[3,1,2]);  % This changes order so we have data, lat, lon
datdiff_u250 = cosweight(datdiff_u250, lat) ; 
datdiff_u250 = permute(datdiff_u250,[2,3,1]); 
datdiff_u250 = double(datdiff_u250); 

%Cosine weight SLP
slp_hpa = permute(slp_hpa,[3,1,2]);  % This changes order so we have u data, lat, lon
slp_hpa = cosweight(slp_hpa, lat) ; 
slp_hpa = permute(slp_hpa,[2,3,1]); 
slp_hpa = double(slp_hpa); 

%Cosine weight 850 temp
datdiff_850t  = permute(datdiff_850t ,[3,1,2]);  % This changes order so we have data, lat, lon
datdiff_850t  = cosweight(datdiff_850t , lat) ; 
datdiff_850t  = permute(datdiff_850t ,[2,3,1]); 
datdiff_850t  = double(datdiff_850t); 



%%  3.  Perform t-test on spatial data
%  3.1   Identify points with 'equal' variance
%  3.2   Calculate t-statistic with pooled std. dev.
%  3.3   Calculate t-statistic with different std. dev.
%  3.4   Identify regions outside of the 95% confidence levels
%  3.5   Plot results



%  We'll use the difference in means quite a bit, so let's go ahead and
%  calculate some stuff...

Np = length(somnum);
Nn = length(slp); 

for j = 1:5
    datdiff1 = squeeze(datdiff(:,:,j)); %dattdiff_u250 %datdiff_850t datdiff
    datdiff2 = squeeze(datdiff_850t(:,:,j));
    datdiff3 = squeeze(slp_hpa(:,:,j));

    clim1 = squeeze(mean(hgt500(:,:,:)));
    clim2 = squeeze(mean(temp850(:,:,:)));
    clim3 = squeeze(mean(slp(:,:,:)));

    stddatp1 = squeeze(std(hgt500(somnum,:,:)));
    stddatn1 = squeeze(std(hgt500(:,:,:)));

    stddatp2 = squeeze(std(temp850(somnum,:,:)));
    stddatn2 = squeeze(std(temp850(:,:,:)));

    stddatp3 = squeeze(std(slp(somnum,:,:)));
    stddatn3 = squeeze(std(slp(:,:,:)));

%  3.1:  Identify points with equal variance
%  1.  90% significance level
%  2.  H0:  variance is the same
%      HA:  Variance is different
%  3.  Statistic:  2-tailed F-test
%  4.  Critical region:  finv(0.05,Np-1,Nn-1) < F < finv(0.95,Np-1,Nn-1)
%  5.  Evaluate the statistic (below)

    for i = 1:3
        eval(['stddatp = stddatp' num2str(i) ' ;'])
        eval(['stddatn = stddatn' num2str(i) ' ;'])
        eval(['datdiff_s = datdiff' num2str(i) ' ;'])

        F = stddatp.^2 ./ stddatn.^2;
        flow = finv(0.05, Np-1, Nn-1);
        fhi = finv(0.95, Np-1, Nn-1);

        samevar = find((flow < F) & (F < fhi));
        diffvar = find((flow > F) | (F > fhi));

        %  3.2  Calculate t-test with pooled std. dev.

        %  Set up t-test with equal variance.
        %  1.  95% significance level
        %  2.  H0:  SST during El Nino events is different than La Nina events
        %      HA:  SST is no different during the two events
        %  3.  Statistic:  2-tailed t-test with N1+N2-2 DOF's
        %  4.  Critical region:  tlow < T < thi
        %  5.  Evaluate the statistic ...


        %  Start by setting up an empty array for the T variable:
        T = repmat(NaN, size(datdiff_s));

        %  Now, calculate t-value for equal variance
        spooled = sqrt(...
            ((Np-1)*stddatp.^2 + (Nn-1)*stddatn.^2) ...
            / (Np + Nn - 2) );
        T(samevar) = datdiff_s(samevar) ./ (spooled(samevar) * sqrt(1/Np + 1/Nn));

        %  3.3  Calculate t-statistic with different std. dev.
        T(diffvar) = datdiff_s(diffvar) ./ ...
            sqrt(stddatp(diffvar).^2/Np + stddatn(diffvar).^2/Nn);

        %  3.4  Identify regions outside the 95% confidence level

        %  Start by identifying DOF's from unequal variance, then replace points
        %  with equal variance with DOF = Np+Nn-2
        DOF = repmat(NaN, size(datdiff_s));
        DOF = ((stddatp.^2/Np) + (stddatn.^2/Nn)).^2 ./ ...
              (((stddatp.^2/Np).^2/(Np-1)) + ((stddatn.^2/Nn).^2/(Nn-1)));
        DOF(samevar) = Np + Nn - 2;
        DOF = round(DOF);

        %  Set up array with 0's where H0 is accepted, and 1's where it is
        %  rejected.
        H = ones(size(datdiff_s));
        H((tinv(0.025, DOF) < T) & (T < tinv(0.975, DOF))) = 0;
        HH(:,:,i,j) = H;

    end
end







%%  PLOTTING 500-hPA GEOPOTENTIAL HEIGHTS AND 250-hPa ZONAL WIND ANOMALIES with 40 m/s CLIMATOLOGICAL ISOTACH
addpath '/Users/mariamadsen/Desktop/Research/tight_subplot'

latmin = 10;
lonmax = 300;
figure(2); 
ha = tight_subplot(5,2,[.01 0],[.01 .01],[.01 .01]);
for i = 1:5
    if i == 1
        axes(ha(i));
    else 
        axes(ha(i+(i-1)))
    end
    
    p = i; 

    u_250_mean = reshape(compwind, [10721, 29, 81]) ;
    u_250_mean = permute(u_250_mean, [1,3,2]) ; 
    u_250_mean = nanmean(u_250_mean) ;
    u_250_mean = squeeze(u_250_mean) ; 
    u_250_mean = permute(u_250_mean, [ 2, 1]) ; 
    
    axesm('eqdcylin','maplatlimit',[latmin latmax], 'maplonlimit',...
        [lonmin lonmax],'meridianlabel', 'on', 'parallellabel', 'on', ...
        'MLabelParallel', 13, 'fontsize', 7, 'fontweight', 'bold', 'FedgeColor', [0.5 0.5 0.5]);

        tightmap; 

        gridm('on'); framem('on'); box('off')
      
           datdiff_geo = squeeze(datdiff(:,:,p));
           datdiff_non = squeeze(datdiff(:,:,p));
           H = squeeze(HH(:,:,1,i)); 
           datdiff_geo(H<1)= NaN; 
           datdiff_non(H==1)= NaN;
       %%%%% 500 hPa GEOPOTENTIAL HEIGHTS (in meters) 

       contourm(latgrat,longrat,datdiff_geo(:,:), 'linecolor', [0.6 0 0], 'LevelList', 25:25:275, 'LineWidth', 1.5);
       hold on 
       contourm(latgrat,longrat,datdiff_geo(:,:), 'linecolor', 'b', 'LevelList', -25:-25:-250, 'LineWidth', 1.5);

%          hold on 
%     
%        contourm(latgrat,longrat,datdiff_non(:,:), 'linecolor', [0.8 0.8 0.8], 'LevelList', 25:25:275, 'LineWidth', 1.5);
%        hold on 
%        contourm(latgrat,longrat,datdiff_non(:,:), 'linecolor', [0.8 0.8 0.8], 'LevelList', -25:-25:-250, 'LineWidth', 1.5);
%        
        hold on
        %%%%%% CLIMATOLOGICAL JET AT 40 m/s

        contourm(latgrat, longrat, u_250_mean, 'Fill', 'off', 'linestyle', '-',...
            'linecolor', [ 0.5    0.5    0.5], 'LineWidth', 4, 'levellist', 40);

        ylim=get(gca,'ylim');
        xlim=get(gca,'xlim');
        set(gca,'ytick',[])

        hold on 

      %%%% ZONAL WIND ANOMALIES 

       contourfm(latgrat,longrat,squeeze(datdiff_u250(:,:,p)), 'LevelList', [-24 -20 -16 -12 -8 -4 4 8 12 16 20 24])%, 'LineStyle', 'none');
       alpha(.1)
       colormap(jetz) ;
       hold on

     %%%%%% TITLE

         title(['LAG: ' num2str(lag(i))], 'FontSize', 25)
         h = get(gca, 'Title');
         set(h, 'Units', 'normalized');
         set(h, 'Position', [0.03,0.8,0.5]);
        set(h, 'HorizontalAlignment', 'left');
         set(h, 'VerticalAlignment', 'bottom');
   
    % Plot the coastlines
    %  Want states?  Try:
    geoshow('usastatehi.shp', 'FaceColor', 'none', 'EdgeColor', 0.5*[1 1 1])

    %  Add a pretty land map
    geoshow('landareas.shp', 'FaceColor', 'none', 'EdgeColor', 'k') ;
        set(gca, 'XColor', [1 1 1], 'YColor', [1 1 1])
        caxis([-30 30]); 
  
end

hold on

% SLP  sig
%PLOTTING 850 TEMPS, and SLP ANOMALIES 


for i = 1:5
    if i == 1
        axes(ha(i+1));
    else 
        axes(ha(i+i))
    end
    
   
    p = i; 
  
    
    axesm('eqdcylin','maplatlimit',[latmin latmax], 'maplonlimit',...
        [lonmin lonmax],'meridianlabel', 'on', 'parallellabel', 'on', ...
        'MLabelParallel', 13, 'fontsize', 7, 'fontweight', 'bold', 'FedgeColor', [0.5 0.5 0.5]);

        tightmap; 

        gridm('on'); framem('on'); box('on')
           
        datdiff_slp = squeeze(slp_hpa(:,:,p));
        H = squeeze(HH(:,:,1,i)); 
        datdiff_slp(H<1)= NaN; 

   %%%%% SEA LEVEL PRESSURE (in PASCALS BUT CHANGED TO HPA ABOVE)
   
    contourm(latgrat,longrat,squeeze(datdiff_slp(:,:)), 'linestyle', '-', ...
       'linecolor', 'k','LineWidth', 2, 'levellist', 3:3:30);


   hold on 
   contourm(latgrat,longrat,squeeze(datdiff_slp(:,:)), 'linestyle', '--', ...
       'linecolor',[0 0 0], 'LineWidth', 2,'levellist', -3:-3:-30);
   
   hold on

 %%%%% 850 Temperature anomaly
        datdiff_T = squeeze(datdiff_850t(:,:,p));
        T = squeeze(HH(:,:,1,i)); 
        datdiff_T(T<1)= NaN; 
    contourfm(latgrat,longrat,squeeze(datdiff_T(:,:)), 'LevelList', [-10 -8 -6 -4 -2 2  4  6  8  10], 'linecolor', [0.5 0.5 0.5]);
       alpha(.2)
       colormap(jetz) ;
       hold on


     %%%%%% TITLE

         title(['LAG: ' num2str(lag(i))], 'FontSize', 25)
         h = get(gca, 'Title');
         set(h, 'Units', 'normalized');
         set(h, 'Position', [0.03,0.8,0.5]);     % Upper left subplot title 
        set(h, 'HorizontalAlignment', 'left');
         set(h, 'VerticalAlignment', 'bottom');
%       % Plot the coastlines
        plotm(lat1, long, 'Color', 'k')
        set(gca, 'XColor', [1 1 1], 'YColor', [1 1 1])
        caxis([-10 10]); 
    hold on
end


%%
jetz = [           0         0    0.5625
         0         0    0.6250
         0         0    0.6875
         0         0    0.7500
         0         0    0.8125
         0         0    0.8750
         0         0    0.9375
         0         0    1.0000
         0    0.0625    1.0000
         0    0.1250    1.0000
         0    0.1875    1.0000
         0    0.2500    1.0000
         0    0.3125    1.0000
         0    0.3750    1.0000
         0    0.4375    1.0000
         0    0.5000    1.0000
         0    0.5625    1.0000
         0    0.6250    1.0000
         0    0.6875    1.0000
         0    0.7500    1.0000
         0    0.8125    1.0000
         0    0.8750    1.0000
         0    0.9375    1.0000
         0    1.0000    1.0000
    0.0625    1.0000    0.9375
    0.1250    1.0000    0.8750
    0.1875    1.0000    0.8125
    0.2500    1.0000    0.7500
    0.3125    1.0000    0.6875
    0.3750    1.0000    0.6250
    1.0000    1.0000    1.0000
    1.0000    1.0000    1.0000
    1.0000    1.0000    1.0000
    0.6250    1.0000    0.3750
    0.6875    1.0000    0.3125
    0.7500    1.0000    0.2500
    0.8125    1.0000    0.1875
    0.8750    1.0000    0.1250
    0.9375    1.0000    0.0625
    1.0000    1.0000         0
    1.0000    0.9375         0
    1.0000    0.8750         0
    1.0000    0.8125         0
    1.0000    0.7500         0
    1.0000    0.6875         0
    1.0000    0.6250         0
    1.0000    0.5625         0
    1.0000    0.5000         0
    1.0000    0.4375         0
    1.0000    0.3750         0
    1.0000    0.3125         0
    1.0000    0.2500         0
    1.0000    0.1875         0
    1.0000    0.1250         0
    1.0000    0.0625         0
    1.0000         0         0
    0.9375         0         0
    0.8750         0         0
    0.8125         0         0
    0.7500         0         0
    0.6875         0         0
    0.6250         0         0
    0.5625         0         0
    0.5000         0         0]; 

