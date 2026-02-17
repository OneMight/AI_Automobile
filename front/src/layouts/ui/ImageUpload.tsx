import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";

interface ImageUploadProps {
  onUpload: (file: File) => void;
}

export const ImageUpload = ({ onUpload }: ImageUploadProps) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onUpload(acceptedFiles[0]);
      }
    },
    [onUpload],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png"],
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={`
        w-full min-h-70 flex flex-col items-center justify-center 
        border-2 border-dashed rounded-lg cursor-pointer transition-colors
        bg-secondary-text/10
        ${isDragActive ? "border-main bg-secondary-bg" : "border-gray-600 hover:border-gray-400"}
      `}
    >
      <input {...getInputProps()} />

      <div className="w-12 h-12 mb-4 rounded-full bg-[#1E293B] flex items-center justify-center">
        <Upload className="text-gray-400 w-6 h-6" />
      </div>

      <p className="text-white text-lg mb-1 text-center">
        Drop image here or click to upload
      </p>

      <p className="text-secondary-text text-sm">JPG, PNG • Max 10MB</p>
    </div>
  );
};
