%% Loading Synoptic Variables 
 clear all
 close all

 
 addpath '/Users/mmmadsen/Desktop/AI2ES/data'
 addpath '/Users/mmmadsen/Desktop/AI2ES/data/Geo500'
 addpath '/Users/mmmadsen/Desktop/AI2ES/data/U250'

% Read in data incrementally

month = [1, 2, 3, 4, 5, 11, 12]; 

z_final = []; 
t_final = []; 
ox = [90:-1.5:15]';   % new latitude to be interpolated
oy = [-180:1.5:180]; % new longitude to be interpolated

for yr = 1979:2019
    s = num2str(yr); 
   disp(['Getting data from ', s ])
    for i = 1:length(month)
        mo = month(i); 
       % geo500 = []; 
        u250 = [];
        %disp(['Getting data from ', s '_' num2str(mo)])
        if mo == 1 || mo ==2 || mo == 3 || mo == 4 || mo == 5
           % filepattern = fullfile(['/Users/mmmadsen/Desktop/AI2ES/data/Geo500/geo500_' ...
           %     s '0' num2str(mo) '.nc']); 
            filepattern = fullfile(['/Users/mmmadsen/Desktop/AI2ES/data/U250/u250_' ...
                s '0' num2str(mo) '.nc']); 
        else
            % filepattern = fullfile(['/Users/mmmadsen/Desktop/AI2ES/data/Geo500/geo500_' ...
            %     s num2str(mo) '.nc']); 
            filepattern = fullfile(['/Users/mmmadsen/Desktop/AI2ES/data/U250/u250_' ...
               s  num2str(mo) '.nc']); 
        end
       % geo = ncread(filepattern, 'z'); 
        u = ncread(filepattern, 'u'); 
        time = ncread(filepattern, 'time'); 
        time = double(time); 
        time = datevec(time/24 + datenum('1900-01-01'));

        z_test = permute(u, [3, 2, 1]); 
        z_shape = reshape(z_test, [4 size(z_test,1)/4 301 1440]); 
        z_mean = squeeze(mean(z_shape)); 

        z_low = permute(z_mean, [3 2 1]); 
      
        for tt = 1:size(z_low, 3)
           % geo500(:,:, tt) = interp2(lon, lat, z_low(:,:, tt)', oy, ox); 
            u250(:,:, tt) = interp2(lon, lat, z_low(:,:, tt)', oy, ox);
        end

        t_daily = time(1:4:end, 1:4); 
   
        z_final = cat(3, z_final, u250); 
        t_final = cat(1, t_final, t_daily); 
    end
end
%% Load in hindcast data Nov-May
 time = ncread('pper_hail_1979_2019.nc', 'time'); 
 time = double(time); 
 time = datevec(time/24 + datenum('1800-01-01'));

 % Only keep months 11, 12, 1, 2, 3, 4, 5, 
Month_find = find(time(:,2)== 1 | time(:,2)== 2 | time(:,2)== 3 | time(:,2)== 4 ...
    | time(:,2)== 5 | time(:,2)== 11 | time(:,2)== 12); 

hail = ncread('pper_hail_1979_2019.nc', 'p_perfect_hail'); 
sig_hail = ncread('pper_sig_hail_1979_2019.nc', 'p_perfect_sig_hail');
tor = ncread('pper_tor_1979_2019.nc', 'p_perfect_tor'); 
sig_tor = ncread('pper_sig_tor_1979_2019.nc', 'p_perfect_sig_tor');
wind = ncread('pper_wind_1979_2019.nc', 'p_perfect_wind'); 
sig_wind = ncread('pper_sig_wind_1979_2019.nc', 'p_perfect_sig_wind');

hail = hail(:,:, Month_find);
sig_hail = sig_hail(:,:,Month_find); 
tor = tor(:,:,Month_find); 
sig_tor = sig_tor(:,:, Month_find); 
wind = wind(:,:,Month_find); 
sig_wind = sig_wind(:,:,Month_find); 

%% Load in OLR 
time_olr = ncread('olr.day.mean.nc', 'time'); 
time_olr = double(time_olr); 
time_olr = datevec(time_olr/24 + datenum('1800-01-01'));
time_olr = time_olr(1676:16650, 1:3);  
olr_find = find(time_olr(:,2)== 1 | time_olr(:,2)== 2 | time_olr(:,2)== 3 | time_olr(:,2)== 4 ...
    | time_olr(:,2)== 5 | time_olr(:,2)== 11 | time_olr(:,2)== 12); 

 

lat_olr = ncread('olr.day.mean.nc', 'lat'); 
lon_olr = ncread('olr.day.mean.nc', 'lon'); 

lat_find=find(lat_olr>=-15 & lat_olr<=15);
lat_olr=lat_olr(lat_find);

olr = ncread('olr.day.mean.nc', 'olr'); 
olr_short = olr(:,lat_find,olr_find) ; 

% Data from code above was saved into mat file named 'synoptic_data.mat' 

%% load in mat file, remove NaNs, remove 21-day running mean, cosweight
load synoptic_data.mat
lat = ox; 
lon = oy'; 

% Set NaN values at last longitude to longitude next to it: 
Geo500_final(:,241,:) = Geo500_final(:,240,:); 
u250_final(:,241,:) = u250_final(:,240,:); 

%% cosweight data: 
Geo500_final = permute(Geo500_final, [3 1 2]); 
u250_final = permute(u250_final, [3 1 2]); 

% cosine weight
addpath '/Users/mmmadsen/Desktop/old_scripts'


Geo500_final = cosweight(Geo500_final, lat) ;
u250_final = cosweight(u250_final, lat) ;


%% Remove leap days from all data now 
leap_ind = find(geo_t_final(:,2)==2 & geo_t_final(:,3) == 29);
Geo500_final(leap_ind, :, :) = []; 
u250_final(leap_ind, :, :) = []; 

u250_t_final(leap_ind, :) = []; 
hail(:,:,leap_ind) = []; 
sig_hail(:,:,leap_ind) = []; 
tor(:,:,leap_ind) = []; 
sig_tor(:,:,leap_ind) = []; 
wind(:,:,leap_ind) = []; 
sig_wind(:,:,leap_ind) = []; 
olr_short(:,:,leap_ind) = []; 




%% Reshape and calculate daily mean 
z_byyear = reshape(Geo500_final, [212 41 51 241]);    % now shaped as day, year, lat, lon
u_byyear = reshape(u250_final, [212 41 51 241]); 

season_meanz = zeros(212, 51, 241);
season_meanu = zeros(212, 51, 241);

% Calculating daily mean across seasons 
for i = 1:212                                                   
    season_meanz(i,:,:) = nanmean(squeeze(z_byyear(i,:,:,:))); 
    season_meanu(i,:,:) = nanmean(squeeze(u_byyear(i,:,:,:))); 
end

% taking 21-day running mean of this
z_day = movmean(season_meanz, 21, 1);                          
u_day = movmean(season_meanu, 21, 1); 


z_season = zeros(212, 41, 51, 241);
u_season = zeros(212, 41, 51, 241);

 % Subtract out this 21-day running mean to get anomalies 
for i = 1:41                               
    for j = 1:212
        z_season(j,i,:,:) = squeeze(z_byyear(j,i,:,:))-squeeze(z_day(j,:,:)); 
        u_season(j,i,:,:) = squeeze(u_byyear(j,i,:,:))-squeeze(u_day(j,:,:)); 

    end
end

% now take a 5-day running mean to smooth data 
z500 = zeros(212, 41, 51, 241);
u250 = zeros(212, 41, 51, 241);
for i = 1:41
    z500(:,i,:,:) = movmean(squeeze(z_season(:,i,:,:)), 5, 1); 
    u250(:,i,:,:) = movmean(squeeze(u_season(:,i,:,:)), 5, 1); 
end


% also- set NaNs as zero 
z500(isnan(z500))=0;
u250(isnan(u250))=0;


%% For olr
olr_short = permute(olr_short, [3 2 1]); 
olr_byyear = reshape(olr_short, [212 41 13 144]); 

season_meanolr = zeros(212, 13, 144);

for i = 1:212                                                   
    season_meanolr(i,:,:) = nanmean(squeeze(olr_byyear(i,:,:,:))); 
end

% taking 21-day running mean of this
olr_day = movmean(season_meanolr, 21, 1);

olr_season = zeros(212, 41, 13, 144);

 % Subtract out this 21-day running mean to get anomalies 
for i = 1:41                               
    for j = 1:212
        olr_season(j,i,:,:) = squeeze(olr_byyear(j,i,:,:))-squeeze(olr_day(j,:,:)); 
    end
end


% now take a 5-day running mean to smooth data 
olr = olr_season; 
% olr = zeros(212, 41, 13, 144);
% 
% for i = 1:41
%     olr(:,i,:,:) = movmean(squeeze(olr_season(:,i,:,:)), 5, 1); 
% end


% also- set NaNs as zero 
olr(isnan(olr))=0;

%% Now reshape so it's all days, lat, lon and save into new mat file with other data 
z500 = reshape(z500, [8692 51 241]); 
u250 = reshape(u250, [8692 51 241]); 
olr = reshape(olr, [8692 13 144]); 
hail = permute(hail, [3 1 2]); 
sig_hail = permute(sig_hail, [3 1 2]);
tor = permute(tor, [3 1 2]);
sig_tor = permute(sig_tor, [3 1 2]);
wind = permute(wind, [3 1 2]);
sig_wind = permute(sig_wind, [3 1 2]);

save('ML_predictors.mat', 'olr', 'lat_olr', 'lon_olr', 'z500', 'u250',...
    'lat', 'lon', 'hail', 'sig_hail', 'tor', 'sig_tor', 'wind', 'sig_wind',...
    'lat_sig', 'lon_sig', 'time'); 



