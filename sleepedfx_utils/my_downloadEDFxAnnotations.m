function my_downloadEDFxAnnotations( varargin )
%downloadEDFxAnnotations Download annotation files from Imperial web server
%   downloadEDFxAnnotations( ) downloads data in the current directory
%   downloadEDFxAnnotations(destination_directory) downloads data in the destination directory


% List of files to download
list_of_files = {
    'hyp_start_time.txt';
    'lights_off_time.txt';
    'lights_on_time.txt';
    'rec_start_time.txt';
    'rec_stop_time.txt'};



% Check if any argument is provided, and if it is, there is only one
% otherwise use the current directory to download data in to
if ~isempty(varargin)
    if length(varargin) > 1
        error('Unknown arguments - the function takes in only one optional argument')
    else
        download_dir = varargin{1};
    end
else
    download_dir = pwd;
end

% Create a destination directory if it doesn't exist
if exist(download_dir, 'dir') ~= 7
    fprintf('WARNING: Download directory does not exist. Creating new directory ...\n\n');
    mkdir(download_dir);
end


%current_dir = pwd;

% URL to download data from
annotations_url = 'https://workspace.imperial.ac.uk/rodriguez-villegas-lab/Public/edfx-toolbox-files/';
% Regular expression to match
regexp_string = '\[t\](........)\[\\t\]';

% Read list of tests from the source url
list_url = [annotations_url 'list_of_tests.txt'];
%edfx_webpage_source = urlread(list_url);
edfx_webpage_source = webread(list_url);
test_names = regexp(edfx_webpage_source,regexp_string,'match');


% Initialise variable for success/failure counts
sc=0;
fc=0;

% Loop through each test to get files
for i=1:length(test_names)
    
    % Add the annotation file specific for each test in the list
    this_test = test_names{i}(4:end-4);
    folder_name = this_test;
    hyp_file = [this_test '.txt'];
    files_to_download = [list_of_files; {hyp_file}];
    
    % Check if test directory exists, create if it doesn't
    test_dir = fullfile(download_dir, folder_name);
    if exist(test_dir,'dir') ~= 7 
        mkdir(download_dir, folder_name);
    end
    
    % Check if info sub-directory exists inside test, create if it doesn't
    info_test_dir = fullfile(test_dir, 'info');
    if exist(info_test_dir,'dir') ~= 7
        mkdir(info_test_dir);
    end
    
    % Download each file from the file_to_download list and display
    % progress and location of saved file
    for f=1:length(files_to_download)
        path_of_file = fullfile(info_test_dir, files_to_download{f});
        url_of_file = [annotations_url this_test '/info/' files_to_download{f}];
        fprintf('Downloading: %s for test %s\n', files_to_download{f}, this_test);
        %[saved_file, status] = urlwrite(url_of_file,path_of_file);
        if(~exist(path_of_file,'file'))
            [saved_file] = websave(path_of_file,url_of_file);
            fprintf('File saved: %s ... OK\n', saved_file);
        else
            disp(['File existed: ', path_of_file]);
        end
%         if status
%             fprintf('File saved: %s ... OK\n', saved_file);
%             sc=sc+1;
%         else
%             fprintf('ERROR DOWNLOADING FILE %s for test %s\n', files_to_download{f}, this_test);
%             fc=fc+1;
%         end
    end
end

% Print final summary of downloads
fprintf('\nDownload complete!\n')
fprintf('\n%d files successfully downloaded ... %d files failed to download\n', sc, fc);

end

