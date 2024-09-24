'use client';

import React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import CloseIcon from '@mui/icons-material/Close';
import PhotoIcon from '@mui/icons-material/Photo';

type FloatingButtonProps = {
  isLoading: boolean;
  handleCancelUpload: () => void;
  handleIconClick: () => void;
};

export default function FloatingButton({
  isLoading,
  handleCancelUpload,
  handleIconClick,
}: FloatingButtonProps) {
  return (
    <div
      className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center shadow-lg cursor-pointer fixed bottom-4 right-4"
      onClick={isLoading ? handleCancelUpload : handleIconClick}
    >
      {isLoading ? (
        <div className="relative">
          <CircularProgress size={32} sx={{ color: '#fff' }} />
          <CloseIcon
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              fontSize: '24px',
              color: '#fff',
              cursor: 'pointer',
            }}
            onClick={handleCancelUpload}
          />
        </div>
      ) : (
        <PhotoIcon className="text-white text-3xl" />
      )}
    </div>
  );
}
