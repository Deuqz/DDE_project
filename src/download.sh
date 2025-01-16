#!/bin/bash

// Get Access: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

rsync -auxvvvL --delete nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_abnormal /home/denis/Загрузки/TUAB
