import numpy as np
import pandas as pd
import xarray as xr


class Train_Test_Val_Split():

    def __init__(self, window, horizon, data_path, pph_path_tor, pph_path_hail, 
                 train_range=['1983-11-01', '2013-05-31'], test_range=['2016-11-01', '2019-05-31'],
                 val_range=['2013-11-01', '2016-05-31']):
        #TODO: change window to be randomized
        self.window = window
        self.horizon = horizon
        self.bins = np.array([0, 24, 24*2, 24*3])

        self.df = xr.load_dataset(data_path)
        self.df['time'] = np.datetime_as_string(self.df['time'])

        self.features = ['u250', 'z500', 'z50', 't800', 'olr']

        self.pph_tor_data = xr.load_dataset(pph_path_tor)
        self.pph_hail_data = xr.load_dataset(pph_path_hail)
        
        self.label_list = list((self.df.time.values))
        for i, val in enumerate(self.label_list):
            self.label_list[i] = val.split('T')[0]
        


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

        train_labels = np.array([])
        for i, val in enumerate(train_range):
            train_labels = np.append(train_labels, self.label_list[self.label_list.index(train_range[i][0]):self.label_list.index(train_range[i][1])])
        
        self.train_labels = train_labels

        test_labels = np.array([])
        for i, val in enumerate(test_range):
            test_labels = np.append(test_labels, self.label_list[self.label_list.index(test_range[i][0]):self.label_list.index(test_range[i][1])])
        
        self.test_labels = test_labels


        val_labels = np.array([])
        for i, val in enumerate(val_range):
            val_labels = np.append(val_labels, self.label_list[self.label_list.index(val_range[i][0]):self.label_list.index(val_range[i][1])])
        
        self.val_labels = val_labels




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
            (len(self.train_labels), self.window, 30, 144, 5))
        
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
        
        a = np.zeros((len(self.train_labels), self.window, 32, 144, 5))   
        a[:, :,:30,:,:] = train_era5_data


        train_pph_data= np.zeros((len(self. train_labels), 65, 93))
        for i, val in enumerate(self.train_labels):
            # train_pph_data[i, :, :, 0] = np.array(self.get_data(val)[2].p_perfect_tor)
            train_pph_data[i, :, :] = np.array(self.get_data(val)[1].p_perfect_hail)
        

        b = np.zeros((len(self.train_labels), 64, 96))
        b[:,:,:93] = train_pph_data[:,:64,:]

        b = np.digitize(b, self.bins)  #temp
        # b = np.eye(4)[b]
        print(np.shape(self.get_data(val)[1].p_perfect_hail))
        print(np.shape(b))
        print(pd.Series(np.ravel(np.mean(b, axis=-1))).value_counts())
        # print(np.shape(b))

        
        val_era5_data = np.zeros(
            (len(self.val_labels), self.window, 30, 144, 5))
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

        c = np.zeros((len(self.val_labels), self.window, 32, 144, 5))   
        c[:, :,:30,:,:] = val_era5_data

        val_pph_data= np.zeros((len(self.val_labels), 65, 93))
        for i, val in enumerate(self.val_labels):
            # val_pph_data[i, :, :, 0] = np.array(self.get_data(val)[2].p_perfect_tor)
            val_pph_data[i, :, :] = np.array(self.get_data(val)[1].p_perfect_hail)
        

        d0 = np.zeros((len(self.val_labels), 64, 96))
        d0[:,:,:93] = val_pph_data[:,:64,:]
        d0 = np.digitize(d0, self.bins)
        # d0 = np.eye(4)[d0]

        print(pd.Series(np.ravel(np.mean(d0, axis=-1))).value_counts())

        test_era5_data = np.zeros(
            (len(self.test_labels), self.window, 30, 144, 5))
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

        e = np.zeros((len(self.test_labels), self.window, 32, 144, 5))   
        e[:, :,:30,:,:] = test_era5_data

        test_pph_data= np.zeros((len(self.test_labels), 65, 93))
        for i, test in enumerate(self.test_labels):
            # test_pph_data[i, :, :, 0] = np.array(self.get_data(test)[2].p_perfect_tor)
            test_pph_data[i, :, :] = np.array(self.get_data(test)[1].p_perfect_hail)

        

        g = np.zeros((len(self.test_labels), 64, 96))
        g[:,:,:93] = test_pph_data[:,:64,:]

        g = np.digitize(g, self.bins)
        # g = np.eye(4)[g]
        print(pd.Series(np.ravel(np.mean(g, axis=-1))).value_counts())

        

        return a, b, c, d0, e, g

    
        
