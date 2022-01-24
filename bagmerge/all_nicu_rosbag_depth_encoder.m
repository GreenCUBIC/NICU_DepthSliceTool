function [] = all_nicu_rosbag_depth_encoder()
    patients = [1, 624; 2, 628; 5, 604; 6, 533; 8, 720; 9, 539; 10, 508;
                11, 535; 13, 585; 14, 705; 15, 650; 16, 615; 17, 550;
                18, 638; 19, 424; 21, 507; 22, 508; 23, 526; 24, 652;
                25, 669; 26, 455; 27, 501; 28, 691; 29, 542; 30, 474;
                31, 503; 32, 504; 33, 618; 34, 506; 35, 574; 36, 496;
                37, 570; 38, 496];
    tic
    parfor i=1:length(patients)
        nicu_rosbag_depth_encoder("Z:\", strcat("D:\DepthVideosEncoded\", "p", num2str(patients(i,1)), "\"), [patients(i, 1)], 0, patients(i, 2))
    end
    toc

end