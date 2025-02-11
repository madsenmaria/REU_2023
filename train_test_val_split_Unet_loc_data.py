import numpy as np
import pandas as pd
import xarray as xr
import random
import teleconnection_data_format


class Train_Test_Val_Split():

    def __init__(self, window, horizon, data_path, pph_path_tor, pph_path_hail, 
                 train_range=['1983-11-01', '2013-05-31'], test_range=['2016-11-01', '2019-05-31'],
                 val_range=['2013-11-01', '2016-05-31']):
        #TODO: change window to be randomized 
        self.window = window
        self.horizon = horizon
        self.bins = np.array([0, 24, 24*2, 24*3])


        # new stuff, old was just self.df = xr.load_dataset(data_path)
        new_data = xr.load_dataset(data_path)
        old_data = xr.open_dataset('/../../../ourdisk/hpc/ai2es/esalm/data/edata_tot.nc')
        new_data = new_data.sel(time=slice('1983-11', '2019-05'))
        new_data = new_data.drop_vars('cin')
        old_data['cape'] = new_data.cape

        self.df = old_data
        self.df['time'] = np.datetime_as_string(self.df['time'])

        self.features = ['u250', 'z500', 'z50', 't800','cape', 'olr']

        self.pph_tor_data = xr.load_dataset(pph_path_tor)
        self.pph_hail_data = xr.load_dataset(pph_path_hail)
        
        self.label_list = list((self.df.time.values))
        for i, val in enumerate(self.label_list):
            self.label_list[i] = val.split('T')[0]
        
        climo = xr.load_dataset('../../../ourdisk/hpc/ai2es/esalm/data/climatology.nc')
        self.climo_train = climo.sel(time=slice(train_range[0], train_range[1]))
        self.climo_test = climo.sel(time=slice(test_range[0], test_range[1]))
        self.climo_val = climo.sel(time=slice(val_range[0], val_range[1]))


        self.global_label_list = [0]*212
        months = ['11', '12', '01', '02', '03', '04', '05']
        days = [30, 31, 31, 28, 31, 30, 31]

        for x, month in enumerate(months):
            for i in range(1, days[x]+1):
                self.global_label_list[(sum(days[:x])+i-1)
                                       ] = (month+'-'+str(i))
                if i < 10:
                    self.global_label_list[(sum(days[:x])+i-1)
                                           ] = (month+'-0'+str(i))

        del months
        del days 

        # train_labels = np.array([])
        # for i, val in enumerate(train_range):
        #     train_labels = np.append(train_labels, self.label_list[self.label_list.index(train_range[i][0]):self.label_list.index(train_range[i][1])])
        
        # self.train_labels = train_labels

        self.train_labels = self.label_list[self.label_list.index(train_range[0]):self.label_list.index(train_range[1])+1]
        self.test_labels = self.label_list[self.label_list.index(test_range[0]):self.label_list.index(test_range[1])+1]
        self.val_labels = self.label_list[self.label_list.index(val_range[0]):self.label_list.index(val_range[1])+1]

        ## Normalizing the data (because I forgor before)

        # need to do this for every variable
        # for feature in self.features:
        #     print(feature)
        #     mu_feature = self.df[feature].mean()
        #     std_feature = self.df[feature].std()
        #     self.df[feature] = (self.df[feature] - mu_feature)/std_feature
        
        # print('sucess')



    def get_window_expanding(self, label=""):
        try:
            # take the season
            # season_index = label_list.index(label)

            # grab all the days previous to one that you have
            window = [0]*(len(self.label_list)-self.horizon)
            for i in range(len(window)):
                window[i] = self.label_list[len(self.label_list)-i-self.horizon-1]
            # return window and label
            return window, label

        # printing error when value is not in seasons list
        except:
            return "Label not found in list"

    # creates a sliding window from given label -- WORKS!
    def get_window_sliding(self, label=""):
        try:
            # ensure not grabbing from diff season
            day_index = self.label_list.index(label)
            m_d = str(label.split('-')[1])+'-'+str(label.split('-')[2])
            global_day_index = self.global_label_list.index(m_d)

            # for values that are almost outside the season, use and expanding window
            if global_day_index < (self.window+self.horizon-1):
                y = label.split('-')[0]
                stopper = y+'-11-01'  # end of the season
                stop_index = self.label_list.index(stopper)

                # creates a new list to grab data from
                label_list_copy = self.label_list.copy()[
                    stop_index:day_index+1]

                return (self.get_window_expanding(label=label, label_list=label_list_copy))

            else:
                window = [0]*(self.window+1)

                for i in range(len(window)):
                    # TODO: 7 days between or pph_day - last_day = 7
                    window[i] = self.label_list[day_index-i-6]
                return window, label

        except:
            return "Label not found in list"

    # returns wind and geopotential dataframe for prev 30 days (minus 7d), olr dataframe for prev 30 days(minus 7d),
    # and pph data for the given date
    def get_data(self, date):
        dates = self.get_window_sliding(date)
        past_dates = dates[0]
        target_data_slicer = slice(date, date)
        if len(past_dates) == 0:
            return [], [], self.pph_hail_data(time=target_data_slicer), self.pph_tor_data(time=target_data_slicer)
        slicer = slice(past_dates[-1], past_dates[0])
        return self.df.sel(time=slicer), self.pph_hail_data.sel(time=target_data_slicer), self.pph_tor_data.sel(time=target_data_slicer)


    def split(self):
        train_era5_data = np.zeros(
            (len(self.train_labels), self.window, 30, 144, 6))
        
        for f, feature in enumerate(self.features):
            for i, val in enumerate(self.train_labels):
                try:
                    m_d = str(val.split('-')[1])+'-'+str(val.split('-')[2])
                    global_day_index = self.global_label_list.index(m_d)

                    if len(self.get_data(val)[0][feature].values) == 0:
                        train_era5_data[i, :, :, :, f] = np.full(
                            (self.window, 30, 144), 0)

                    # for values that are almost outside the season, use and expanding window
                    elif global_day_index < (self.window+self.horizon):
                        r = np.zeros((self.window, 30, 144))
                        n = self.get_data(val)[0][feature].values
                        n = np.transpose(n, [0, 2, 1])
                        r[0:len(n), :, :] = n
                        r[len(n):, :, :] = np.full(
                            ((self.window-len(n)), 30, 144), 0)
                        train_era5_data[i, :, :, :, f] = r
                    else:
                        d = self.get_data(val)[0][feature].values
                        d = np.transpose(d, [0, 2, 1])
                        train_era5_data[i, :, :, :, f] = d
                except Exception as e:
                    train_era5_data[i, :, :, :, f] = np.full(
                        (self.window, 30, 144), 0)
        
        a = np.zeros((len(self.train_labels), self.window, 32, 144, 6))   
        a[:, :,:30,:,:] = train_era5_data


        train_pph_data= np.zeros((len(self. train_labels), 65, 93, 2))
        for i, val in enumerate(self.train_labels):
            train_pph_data[i, :, :, 0] = np.array(self.get_data(val)[2].p_perfect_tor)
            train_pph_data[i, :, :, 1] = np.array(self.get_data(val)[1].p_perfect_hail)
        

        b = np.zeros((len(self.train_labels), 64, 96, 1))
        b[:,:,:93, :] = train_pph_data[:,:64,:, 1].reshape(train_pph_data.shape[0],64,93,1)
        

        climo_o_train = np.zeros((len(self.train_labels), 64, 96))
        climo_o_train[:, :, :93] = self.climo_train.hail.values[:, :64, :]
    

        # climo_o_train = np.concatenate([climo_o_train, climo_o_train[tl, :, :]])


        
        
        val_era5_data = np.zeros(
            (len(self.val_labels), self.window, 30, 144, 6))
        for f, feature in enumerate(self.features):
            for i, val in enumerate(self.val_labels):
                try:
                    m_d = str(val.split('-')[1])+'-'+str(val.split('-')[2])
                    global_day_index = self.global_label_list.index(m_d)

                    if len(self.get_data(val)[0][feature].values) == 0:
                        val_era5_data[i, :, :, :, f] = np.full(
                            (self.window, 30, 144), 0)

                    # for values that are almost outside the season, use and expanding window
                    elif global_day_index < (self.window+self.horizon):
                        r = np.zeros((self.window, 30, 144))
                        n = self.get_data(val)[0][feature].values
                        n = np.transpose(n, [0, 2, 1])
                        r[0:len(n), :, :] = n
                        r[len(n):, :, :] = np.full(
                            ((self.window-len(n)), 30, 144), 0)
                        val_era5_data[i, :, :, :, f] = r
                    else:
                        d = self.get_data(val)[0][feature].values
                        d = np.transpose(d, [0, 2, 1])
                        val_era5_data[i, :, :, :, f] = d
                except Exception as e:
                    val_era5_data[i, :, :, :, f] = np.full(
                        (self.window, 30, 144), 0)

        c = np.zeros((len(self.val_labels), self.window, 32, 144, 6))   
        c[:, :,:30,:,:] = val_era5_data

        val_pph_data= np.zeros((len(self.val_labels), 65, 93, 2))
        for i, val in enumerate(self.val_labels):
            val_pph_data[i, :, :, 0] = np.array(self.get_data(val)[2].p_perfect_tor)
            val_pph_data[i, :, :, 1] = np.array(self.get_data(val)[1].p_perfect_hail)
        

        d0 = np.zeros((len(self.val_labels), 64, 96, 1))
        d0[:,:,:93, :] = val_pph_data[:,:64,:, 1].reshape(636,64,93,1)
    
        climo_o_val = np.zeros((len(self.val_labels), 64, 96))
        climo_o_val[:, :, :93] = self.climo_val.hail.values[:, :64, :]

    

        test_era5_data = np.zeros(
            (len(self.test_labels), self.window, 30, 144, 6))
        for f, feature in enumerate(self.features):
            for i, test in enumerate(self.test_labels):
                try:
                    m_d = str(test.split('-')[1])+'-'+str(test.split('-')[2])
                    global_day_index = self.global_label_list.index(m_d)

                    if len(self.get_data(test)[0][feature].values) == 0:
                        test_era5_data[i, :, :, :, f] = np.full(
                            (self.window, 30, 144), 0)

                    # for values that are almost outside the season, use and expanding window
                    elif global_day_index < (self.window+self.horizon):
                        r = np.zeros((self.window, 30, 144))
                        n = self.get_data(test)[0][feature].values
                        n = np.transpose(n, [0, 2, 1])
                        r[0:len(n), :, :] = n
                        r[len(n):, :, :] = np.full(
                            ((self.window-len(n)), 30, 144), 0)
                        test_era5_data[i, :, :, :, f] = r
                    else:
                        d = self.get_data(test)[0][feature].values
                        d = np.transpose(d, [0, 2, 1])
                        test_era5_data[i, :, :, :, f] = d
                except Exception as e:
                    test_era5_data[i, :, :, :, f] = np.full(
                        (self.window, 30, 144), 0)

        print('test_era5 data shape before placed in e')
        print(np.shape(test_era5_data))
        e = np.zeros((len(self.test_labels), self.window, 32, 144, 6))   
        e[:, :,:30,:,:] = test_era5_data

        test_pph_data= np.zeros((len(self.test_labels), 65, 93, 2))
        for i, test in enumerate(self.test_labels):
            test_pph_data[i, :, :, 0] = np.array(self.get_data(test)[2].p_perfect_tor)
            test_pph_data[i, :, :, 1] = np.array(self.get_data(test)[1].p_perfect_hail)

        g = np.zeros((len(self.test_labels), 64, 96, 1))
        g[:,:,:93, :] = test_pph_data[:,:64,:, 1].reshape(636,64,93,1)

        climo_o_test = np.zeros((len(self.test_labels), 64, 96))
        climo_o_test[:, :, :93] = self.climo_test.hail.values[:, :64, :]

        full_tele_data = teleconnection_data_format.get_teleconnection_data()
        
        return a, b, c, d0, e, g, climo_o_train, climo_o_val, climo_o_test, full_tele_data[:(30*212), :], full_tele_data[(30*212):(33*212), :], full_tele_data[(33*212):, :]

    
        
