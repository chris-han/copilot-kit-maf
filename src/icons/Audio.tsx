import { Volume2 as LucideVolume2, LucideProps } from 'lucide-react';

const Audio = ({ className, ...props }: LucideProps) => {
  return <LucideVolume2 className={className} {...props} />;
};

export default Audio;