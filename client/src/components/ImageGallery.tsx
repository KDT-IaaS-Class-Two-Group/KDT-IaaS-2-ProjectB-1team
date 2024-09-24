'use client';

import React from 'react';

type ImageGalleryProps = {
  images: string[];
};

export default function ImageGallery({ images }: ImageGalleryProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-8">
      {images.map((image, index) => (
        <div key={index} className="relative pb-[56.25%] h-0">
          <img
            src={image}
            alt={`uploaded-${index}`}
            className="absolute top-0 left-0 w-full h-full object-cover rounded-lg"
          />
        </div>
      ))}
    </div>
  );
}
