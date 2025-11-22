import { Download as LucideDownload, LucideProps } from 'lucide-react';

const Download = ({ className, ...props }: LucideProps) => {
  return <LucideDownload className={className} {...props} />;
};

export default Download;