import { Phone as LucidePhone, LucideProps } from 'lucide-react';

const CallIcon = ({ className, ...props }: LucideProps) => {
  return <LucidePhone className={className} {...props} />;
};

export default CallIcon;